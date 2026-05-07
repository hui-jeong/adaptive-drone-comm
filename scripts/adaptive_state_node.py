import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import csv
import math
import os
import time


OBSTACLE_X = -1.6
OBSTACLE_Y = 570.0

T_CAUTION = 3.0
T_EMERGENCY = 1.5
D_MARGIN = 1.0


class AdaptiveStateNode(Node):

    def __init__(self):
        super().__init__('adaptive_state_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.listener_callback,
            qos_profile
        )

        os.makedirs("logs", exist_ok=True)
        self.file = open("logs/adaptive_real_log.csv", "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "time", "x", "y", "vx", "vy",
            "distance", "v_close",
            "d_caution", "d_emergency",
            "state"
        ])

        self.start_time = time.time()

    def listener_callback(self, msg):
        x = msg.x
        y = msg.y
        vx = msg.vx
        vy = msg.vy

        dx = OBSTACLE_X - x
        dy = OBSTACLE_Y - y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > 0:
            unit_x = dx / distance
            unit_y = dy / distance
            v_close = vx * unit_x + vy * unit_y
        else:
            v_close = 0.0

        v_close = max(v_close, 0.0)

        d_caution = v_close * T_CAUTION + D_MARGIN
        d_emergency = v_close * T_EMERGENCY + D_MARGIN

        if distance <= d_emergency:
            state = "EMERGENCY"
        elif distance <= d_caution:
            state = "CAUTION"
        else:
            state = "NORMAL"

        t = time.time() - self.start_time

        self.writer.writerow([
            round(t, 2),
            round(x, 3),
            round(y, 3),
            round(vx, 3),
            round(vy, 3),
            round(distance, 3),
            round(v_close, 3),
            round(d_caution, 3),
            round(d_emergency, 3),
            state
        ])
        self.file.flush()

        print(
            f"x={x:.2f}, y={y:.2f}, "
            f"dist={distance:.2f}, "
            f"v_close={v_close:.2f}, "
            f"state={state}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.file.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()