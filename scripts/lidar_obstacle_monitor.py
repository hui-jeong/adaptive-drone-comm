import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import math


SCAN_TOPIC = "/world/default/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan"


class LidarMonitor(Node):

    def __init__(self):
        super().__init__('lidar_monitor')

        self.subscription = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        min_distance = 999.0
        min_angle = 0.0

        front_min = 999.0
        front_center_min = 999.0

        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_increment
            angle_deg = math.degrees(angle)

            if math.isinf(r) or math.isnan(r) or r < 0.05:
                continue

            # 전체 LiDAR에서 가장 가까운 물체
            if r < min_distance:
                min_distance = r
                min_angle = angle_deg

            # 기존 전방 범위: -30도 ~ +30도
            if -30.0 <= angle_deg <= 30.0:
                front_min = min(front_min, r)
            # 참고용: +90도 근처도 확인
            if 60.0 <= angle_deg <= 120.0:
                front_center_min = min(front_center_min, r)

        if min_distance == 999.0:
            print("No valid LiDAR data")
            return

        if front_min < 1.5:
            front_state = "EMERGENCY"
        elif front_min < 3.0:
            front_state = "CAUTION"
        else:
            front_state = "NORMAL"

        print(
            f"Closest: {min_distance:.2f} m | "
            f"Angle: {min_angle:.1f} deg | "
            f"Front(-30~30): {front_min:.2f} m | "
            f"State: {front_state} | "
            f"Side(+60~120): {front_center_min:.2f} m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()