import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from px4_msgs.msg import (
    VehicleLocalPosition,
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand
)


SCAN_TOPIC = "/world/default/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan"

TARGET_ALTITUDE = -2.5

# 목표점: 장애물 뒤쪽까지 계속 직진
GOAL_X = 0.0
GOAL_Y = 80.0

T_CAUTION = 3.0
T_EMERGENCY = 1.5
D_MARGIN = 1.0
MAX_V_CLOSE = 3.0

LOOKAHEAD_DISTANCE = 2.0
AVOID_SIDE_DISTANCE = 5.0
RETURN_GAIN = 0.15
PATH_TOLERANCE = 0.5

DIRECTION_HOLD_TIME = 1.0


class VFHDroneControl(Node):

    def __init__(self):
        super().__init__("vfh_drone_control")

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self.scan_callback,
            10
        )

        self.pos_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.position_callback,
            px4_qos
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10
        )

        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10
        )

        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10
        )

        self.x = 0.0
        self.y = 0.0
        self.z = TARGET_ALTITUDE

        self.path_initialized = False
        self.path_start_x = 0.0
        self.path_start_y = 0.0

        self.path_dir_x = 0.0
        self.path_dir_y = 1.0
        self.path_left_x = -1.0
        self.path_left_y = 0.0

        self.front_min = 999.0
        self.left_avg = 0.0
        self.right_avg = 0.0

        self.prev_front_min = None
        self.prev_scan_time = None
        self.v_close = 0.0

        self.mode = "FOLLOW_PATH"
        self.avoid_direction = "LEFT"
        self.last_direction_change_time = time.time()

        self.offboard_counter = 0

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.avoid_start_time = None
        self.flight_start_time = None
        self.return_stable_start_time = None

    def position_callback(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.z = msg.z

        if not self.path_initialized:
            self.path_start_x = self.x
            self.path_start_y = self.y

            dx = GOAL_X - self.path_start_x
            dy = GOAL_Y - self.path_start_y
            norm = math.sqrt(dx * dx + dy * dy)

            self.path_dir_x = dx / norm
            self.path_dir_y = dy / norm

            self.path_left_x = -self.path_dir_y
            self.path_left_y = self.path_dir_x

            self.path_initialized = True

            print(
                f"Path initialized | "
                f"start=({self.path_start_x:.2f}, {self.path_start_y:.2f}) | "
                f"goal=({GOAL_X:.2f}, {GOAL_Y:.2f}) | "
                f"path_dir=({self.path_dir_x:.2f}, {self.path_dir_y:.2f})"
            )

    def scan_callback(self, msg):
        front = []
        left = []
        right = []

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_inc

            if math.isinf(r) or math.isnan(r) or r < 1.0:
                continue

            if -math.radians(30) <= angle <= math.radians(30):
                front.append(r)

            elif math.radians(30) < angle <= math.radians(90):
                left.append(r)

            elif -math.radians(90) <= angle < -math.radians(30):
                right.append(r)

        if not front:
            self.front_min = 999.0
            self.v_close = 0.0
            self.prev_front_min = None
            self.prev_scan_time = None
            return

        now = time.time()

        self.front_min = min(front)
        self.left_avg = sum(left) / len(left) if left else 0.0
        self.right_avg = sum(right) / len(right) if right else 0.0

       
        # 초기 LiDAR 값 안정화
        # 첫 측정값은 v_close 계산에 사용하지 않음
        if self.prev_front_min is None or self.prev_scan_time is None:
            self.prev_front_min = self.front_min
            self.prev_scan_time = now
            self.v_close = 0.0
            return

        dt = now - self.prev_scan_time

        if dt > 0:
            raw_v_close = (self.prev_front_min - self.front_min) / dt

            # 멀어지는 중이면 접근속도 0
            raw_v_close = max(raw_v_close, 0.0)

            # 초기 노이즈/순간 튐 제거
            if raw_v_close > 5.0:
                raw_v_close = 0.0

            # 최대 접근속도 제한
            raw_v_close = min(raw_v_close, MAX_V_CLOSE)

            # 지수 이동 평균 필터
            alpha = 0.7
            self.v_close = alpha * self.v_close + (1.0 - alpha) * raw_v_close

        self.prev_front_min = self.front_min
        self.prev_scan_time = now

    def get_path_error(self):
        dx = self.x - self.path_start_x
        dy = self.y - self.path_start_y
        return dx * self.path_left_x + dy * self.path_left_y

    def get_projection_point(self):
        dx = self.x - self.path_start_x
        dy = self.y - self.path_start_y

        s = dx * self.path_dir_x + dy * self.path_dir_y

        proj_x = self.path_start_x + s * self.path_dir_x
        proj_y = self.path_start_y + s * self.path_dir_y

        return proj_x, proj_y, s

    def update_mode(self):
        d_caution = max(self.v_close * T_CAUTION + D_MARGIN, 2.0)
        d_emergency = max(self.v_close * T_EMERGENCY + D_MARGIN, 1.5)

        # enter_avoid = (self.front_min <= 12.0) and (self.v_close > 0.5)
        enter_avoid = self.front_min <= 6.0
        # AVOID에 들어온 뒤 최소 유지 시간
        MIN_AVOID_TIME = 2.0

        # 장애물을 충분히 지나갔다고 보는 조건
        clear_obstacle = (
            self.front_min >= 18.0 and
            self.v_close < 0.2 and
            abs(self.get_path_error()) >= 2.5
        )

        now = time.time()

        if self.mode == "FOLLOW_PATH":
            if enter_avoid:
                self.mode = "AVOID"
                self.avoid_direction = "LEFT"
                self.avoid_start_time = now
                _, _, self.avoid_start_s = self.get_projection_point()

        elif self.mode == "AVOID":
            avoid_elapsed = now - getattr(self, "avoid_start_time", now)
            _, _, current_s = self.get_projection_point()
            avoid_progress = current_s - getattr(self, "avoid_start_s", current_s)

            clear_obstacle = (
                avoid_elapsed >= 6.0 and
                avoid_progress >= 12.0 and
                abs(self.get_path_error()) >= 4.0
            )

            if clear_obstacle:
                self.mode = "RETURN_PATH"
            # if avoid_elapsed >= MIN_AVOID_TIME and clear_obstacle:
            #     self.mode = "RETURN_PATH"

        elif self.mode == "RETURN_PATH":
            path_error = abs(self.get_path_error())

            if path_error < PATH_TOLERANCE:
                if self.return_stable_start_time is None:
                    self.return_stable_start_time = now
                elif now - self.return_stable_start_time >= 1.0:
                    self.mode = "FOLLOW_PATH"
                    self.return_stable_start_time = None
            else:
                self.return_stable_start_time = None

            # if enter_avoid:
            #     self.mode = "AVOID"
            #     self.avoid_direction = "LEFT"
            #     self.avoid_start_time = now
            # elif path_error < PATH_TOLERANCE:
            #     self.mode = "FOLLOW_PATH"

        return d_caution, d_emergency

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def timer_callback(self):
        timestamp = int(self.get_clock().now().nanoseconds / 1000)

        offboard = OffboardControlMode()
        offboard.timestamp = timestamp
        offboard.position = True
        offboard.velocity = False
        offboard.acceleration = False
        offboard.attitude = False
        offboard.body_rate = False
        self.offboard_pub.publish(offboard)

        self.control(timestamp)

        if self.offboard_counter == 100:
            print("Switching to OFFBOARD")
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                1.0,
                6.0
            )

        if self.offboard_counter == 120:
            print("Arming")
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                1.0
            )

        self.offboard_counter += 1

    def control(self, timestamp):
        if not self.path_initialized:
            return

        d_caution, d_emergency = self.update_mode()

        proj_x, proj_y, s = self.get_projection_point()
        path_error = self.get_path_error()

        base_x = proj_x + self.path_dir_x * LOOKAHEAD_DISTANCE
        base_y = proj_y + self.path_dir_y * LOOKAHEAD_DISTANCE

        if self.mode == "FOLLOW_PATH":
            target_x = base_x
            target_y = base_y

        elif self.mode == "AVOID":
            side_sign = 1.0 if self.avoid_direction == "LEFT" else -1.0
            target_x = base_x + self.path_left_x * AVOID_SIDE_DISTANCE * side_sign
            target_y = base_y + self.path_left_y * AVOID_SIDE_DISTANCE * side_sign

        elif self.mode == "RETURN_PATH":
            target_x = base_x + self.path_left_x * path_error * RETURN_GAIN
            target_y = base_y + self.path_left_y * path_error * RETURN_GAIN

        else:
            target_x = base_x
            target_y = base_y

        sp = TrajectorySetpoint()
        sp.timestamp = timestamp
        sp.position = [
            float(target_x),
            float(target_y),
            float(TARGET_ALTITUDE)
        ]

        sp.yaw = float(math.atan2(self.path_dir_y, self.path_dir_x))
        self.traj_pub.publish(sp)

        print(
            f"mode={self.mode} | "
            f"front={self.front_min:.2f} | "
            f"v_close={self.v_close:.2f} | "
            f"d_caution={d_caution:.2f} | "
            f"d_emergency={d_emergency:.2f} | "
            f"path_error={path_error:.2f} | "
            f"avoid={self.avoid_direction}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = VFHDroneControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()