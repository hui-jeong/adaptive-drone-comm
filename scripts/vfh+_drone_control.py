import math
import time
from dataclasses import dataclass
from typing import List, Tuple

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

# =========================
# Altitude / Takeoff
# =========================

# PX4 local z가 0이 아닐 수 있으므로 home_z + TARGET_ALTITUDE 로 target_z를 만든다.
TARGET_ALTITUDE = -2.5
TAKEOFF_ALTITUDE_TOLERANCE = 0.3
TAKEOFF_STABLE_DURATION = 1.0

# =========================
# Path
# =========================

GOAL_X = 0.0
GOAL_Y = 80.0

PATH_TOLERANCE = 0.5

# =========================
# VFH+ / Mode transition
# =========================

VFH_ENTER_AVOID_DISTANCE = 6.0

# AVOID -> RETURN_PATH 전환 기준
MIN_AVOID_TIME = 1.5
CLEAR_FRONT_DISTANCE = 8.0
CLEAR_SIDE_DISTANCE = 4.5
CLEAR_COUNT_THRESHOLD = 10

# RETURN_PATH 중 다시 AVOID로 들어가는 기준
RETURN_REENTER_DISTANCE = 6.0
RETURN_REENTER_SIDE_DISTANCE = 5.0

# heading smoothing용 최소 회피각
MIN_HEADING_FOR_SIDE_DECISION_DEG = 8.0

# =========================
# Target generation
# =========================

FOLLOW_TARGET_DISTANCE = 2.5
AVOID_TARGET_DISTANCE = 3.0
RETURN_TARGET_DISTANCE = 3.0

# =========================
# Velocity control
# =========================

FOLLOW_SPEED = 0.7
AVOID_SPEED = 0.55
RETURN_SPEED = 0.45

MAX_XY_SPEED = 0.7
MAX_Z_SPEED = 0.30
MAX_VEL_STEP = 0.05

ALTITUDE_GAIN = 0.6


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_diff(a: float, b: float) -> float:
    return abs(normalize_angle(a - b))


@dataclass
class VFHPlusConfig:
    max_range: float = 10.0

    # 6m 이내 장애물만 VFH+ histogram에서 위험 영역으로 반영
    active_distance: float = 6.0

    robot_radius: float = 0.35
    safety_margin: float = 1.0

    sector_deg: float = 5.0
    front_angle_deg: float = 30.0
    max_steer_deg: float = 80.0

    weight_target: float = 5.0
    weight_current: float = 2.0
    weight_previous: float = 5.0


class VFHPlusPlanner:
    def __init__(self, config: VFHPlusConfig):
        self.cfg = config
        self.sector_rad = math.radians(self.cfg.sector_deg)
        self.num_sectors = int(round(2.0 * math.pi / self.sector_rad))
        self.prev_selected_heading = 0.0

    def angle_to_index(self, angle: float) -> int:
        angle = normalize_angle(angle)
        idx = int((angle + math.pi) / self.sector_rad)
        return max(0, min(self.num_sectors - 1, idx))

    def index_to_angle(self, idx: int) -> float:
        return normalize_angle(-math.pi + (idx + 0.5) * self.sector_rad)

    def circular_index_distance(self, a: int, b: int) -> int:
        d = abs(a - b)
        return min(d, self.num_sectors - d)

    def get_front_min_distance(self, scan: LaserScan) -> float:
        front_half_angle = math.radians(self.cfg.front_angle_deg)
        min_dist = self.cfg.max_range

        angle = scan.angle_min

        for r in scan.ranges:
            norm_angle = normalize_angle(angle)

            if -front_half_angle <= norm_angle <= front_half_angle:
                if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                    min_dist = min(min_dist, r)

            angle += scan.angle_increment

        return min_dist

    def build_blocked_histogram(self, scan: LaserScan) -> List[bool]:
        min_ranges = [self.cfg.max_range for _ in range(self.num_sectors)]

        angle = scan.angle_min

        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                r = min(r, self.cfg.max_range)
                idx = self.angle_to_index(angle)
                min_ranges[idx] = min(min_ranges[idx], r)

            angle += scan.angle_increment

        blocked = [False for _ in range(self.num_sectors)]
        clearance = self.cfg.robot_radius + self.cfg.safety_margin

        for idx, dist in enumerate(min_ranges):
            if dist >= self.cfg.active_distance:
                continue

            if dist <= clearance:
                expand_angle = math.pi / 2.0
            else:
                expand_angle = math.asin(min(1.0, clearance / dist))

            expand_sector = int(math.ceil(expand_angle / self.sector_rad))

            for offset in range(-expand_sector, expand_sector + 1):
                blocked[(idx + offset) % self.num_sectors] = True

        return blocked

    def find_valleys(self, blocked: List[bool]) -> List[Tuple[int, int, int]]:
        n = self.num_sectors
        free = [not b for b in blocked]

        if not any(free):
            return []

        if all(free):
            return [(0, n - 1, n)]

        valleys = []

        for i in range(n):
            prev_i = (i - 1) % n

            if free[i] and not free[prev_i]:
                start = i
                length = 0
                j = i

                while free[j % n] and length < n:
                    length += 1
                    j += 1

                end = (j - 1) % n
                valleys.append((start, end, length))

        return valleys

    def index_in_valley(self, idx: int, start: int, end: int) -> bool:
        if start <= end:
            return start <= idx <= end

        return idx >= start or idx <= end

    def get_candidate_from_valley(
        self,
        valley: Tuple[int, int, int],
        target_idx: int
    ) -> int:
        start, end, width = valley

        if self.index_in_valley(target_idx, start, end):
            return target_idx

        margin = 1

        left_edge = (start + margin) % self.num_sectors
        right_edge = (end - margin) % self.num_sectors

        left_dist = self.circular_index_distance(left_edge, target_idx)
        right_dist = self.circular_index_distance(right_edge, target_idx)

        if left_dist <= right_dist:
            return left_edge
        else:
            return right_edge

    def cost(
        self,
        candidate_heading: float,
        target_heading: float,
        current_heading: float,
        previous_heading: float
    ) -> float:
        return (
            self.cfg.weight_target * angle_diff(candidate_heading, target_heading)
            + self.cfg.weight_current * angle_diff(candidate_heading, current_heading)
            + self.cfg.weight_previous * angle_diff(candidate_heading, previous_heading)
        )

    def select_heading(
        self,
        scan: LaserScan,
        target_heading: float = 0.0,
        current_heading: float = 0.0
    ) -> Tuple[float, str, float]:
        front_min = self.get_front_min_distance(scan)

        if front_min > VFH_ENTER_AVOID_DISTANCE:
            self.prev_selected_heading = 0.0
            return 0.0, "CRUISE", front_min

        blocked = self.build_blocked_histogram(scan)
        valleys = self.find_valleys(blocked)

        max_steer = math.radians(self.cfg.max_steer_deg)

        if not valleys:
            if self.prev_selected_heading < 0.0:
                fallback = -max_steer
            elif self.prev_selected_heading > 0.0:
                fallback = max_steer
            else:
                fallback = -max_steer

            self.prev_selected_heading = fallback
            return fallback, "BLOCKED", front_min

        target_idx = self.angle_to_index(target_heading)
        candidates = []

        for valley in valleys:
            candidate_idx = self.get_candidate_from_valley(valley, target_idx)
            candidate_angle = self.index_to_angle(candidate_idx)

            if abs(candidate_angle) <= max_steer:
                candidates.append(candidate_angle)

        if not candidates:
            if self.prev_selected_heading < 0.0:
                fallback = -max_steer
            elif self.prev_selected_heading > 0.0:
                fallback = max_steer
            else:
                fallback = -max_steer

            self.prev_selected_heading = fallback
            return fallback, "BLOCKED", front_min

        best_heading = min(
            candidates,
            key=lambda h: self.cost(
                h,
                target_heading,
                current_heading,
                self.prev_selected_heading
            )
        )

        best_heading = max(-max_steer, min(max_steer, best_heading))

        self.prev_selected_heading = best_heading
        return best_heading, "AVOID", front_min


class VFHPlusDroneControl(Node):

    def __init__(self):
        super().__init__("vfh_plus_drone_control")

        self.vfh_config = VFHPlusConfig()
        self.vfh_planner = VFHPlusPlanner(self.vfh_config)

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
        self.z = 0.0

        self.home_z = None
        self.target_z = None

        self.path_initialized = False
        self.path_start_x = 0.0
        self.path_start_y = 0.0

        self.path_dir_x = 0.0
        self.path_dir_y = 1.0
        self.path_left_x = -1.0
        self.path_left_y = 0.0

        self.front_min = self.vfh_config.max_range

        self.left_avg = 0.0
        self.right_avg = 0.0

        self.left_min = self.vfh_config.max_range
        self.right_min = self.vfh_config.max_range
        self.side_min = self.vfh_config.max_range

        self.prev_front_min = None
        self.prev_scan_time = None
        self.v_close = 0.0

        self.mode = "FOLLOW_PATH"

        self.avoid_sign = 0.0
        # 이륙 안정화 및 회피 활성화 gate
        self.takeoff_stable = False
        self.takeoff_stable_start_time = None
        self.avoidance_enabled = False

        self.vfh_selected_heading = 0.0
        self.vfh_planner_state = "CRUISE"

        self.smoothed_vfh_heading = 0.0
        self.prev_smoothed_vfh_heading = 0.0
        self.last_nonzero_vfh_heading = 0.0

        self.front_clear_count = 0
        self.avoid_elapsed_for_log = 0.0
        self.avoid_progress_for_log = 0.0

        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_vz = 0.0

        self.offboard_counter = 0

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.avoid_start_time = None
        self.avoid_start_s = 0.0
        self.return_stable_start_time = None

        self.get_logger().info(
            "Simplified VFH+ velocity control node started."
        )

    def position_callback(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.z = msg.z

        if self.home_z is None:
            self.home_z = self.z
            self.target_z = self.home_z + TARGET_ALTITUDE

            print(
                f"Altitude reference initialized | "
                f"home_z={self.home_z:.2f} | "
                f"target_z={self.target_z:.2f}"
            )

        if not self.path_initialized:
            self.path_start_x = self.x
            self.path_start_y = self.y

            dx = GOAL_X - self.path_start_x
            dy = GOAL_Y - self.path_start_y
            norm = math.sqrt(dx * dx + dy * dy)

            if norm < 1e-6:
                self.get_logger().warn(
                    "Goal is too close to start position. Using default path direction."
                )
                self.path_dir_x = 0.0
                self.path_dir_y = 1.0
            else:
                self.path_dir_x = dx / norm
                self.path_dir_y = dy / norm

            self.path_left_x = -self.path_dir_y
            self.path_left_y = self.path_dir_x

            self.path_initialized = True

            print(
                f"Path initialized | "
                f"start=({self.path_start_x:.2f}, {self.path_start_y:.2f}) | "
                f"goal=({GOAL_X:.2f}, {GOAL_Y:.2f}) | "
                f"path_dir=({self.path_dir_x:.2f}, {self.path_dir_y:.2f}) | "
                f"path_left=({self.path_left_x:.2f}, {self.path_left_y:.2f})"
            )

    def scan_callback(self, msg):
        selected_heading, planner_state, vfh_front_min = self.vfh_planner.select_heading(
            msg,
            target_heading=0.0,
            current_heading=self.smoothed_vfh_heading
        )

        self.vfh_selected_heading = selected_heading
        self.vfh_planner_state = planner_state

        if abs(selected_heading) >= math.radians(MIN_HEADING_FOR_SIDE_DECISION_DEG):
            self.last_nonzero_vfh_heading = selected_heading

        front = []
        left = []
        right = []

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        for i, r in enumerate(msg.ranges):
            angle = normalize_angle(angle_min + i * angle_inc)

            if not math.isfinite(r):
                continue

            if r < msg.range_min or r > msg.range_max:
                continue

            if -math.radians(30) <= angle <= math.radians(30):
                front.append(r)

            elif math.radians(30) < angle <= math.radians(120):
                left.append(r)

            elif -math.radians(120) <= angle < -math.radians(30):
                right.append(r)

        if front:
            self.front_min = min(front)
        else:
            self.front_min = vfh_front_min

        self.left_avg = sum(left) / len(left) if left else 0.0
        self.right_avg = sum(right) / len(right) if right else 0.0

        self.left_min = min(left) if left else self.vfh_config.max_range
        self.right_min = min(right) if right else self.vfh_config.max_range
        self.side_min = min(self.left_min, self.right_min)

        now = time.time()

        if self.prev_front_min is None or self.prev_scan_time is None:
            self.prev_front_min = self.front_min
            self.prev_scan_time = now
            self.v_close = 0.0
            return

        dt = now - self.prev_scan_time

        if dt > 0:
            raw_v_close = (self.prev_front_min - self.front_min) / dt
            raw_v_close = max(raw_v_close, 0.0)

            # 순간적인 LiDAR 튐 제거
            if raw_v_close > 5.0:
                raw_v_close = 0.0

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

    def update_takeoff_stable_state(self, now: float):
        """
        목표 고도 근처에 일정 시간 머무르면 이륙 안정화가 완료된 것으로 판단한다.
        """
        if self.takeoff_stable:
            return

        if self.target_z is None:
            self.takeoff_stable = False
            self.takeoff_stable_start_time = None
            return

        altitude_error = abs(self.target_z - self.z)
        altitude_ready = altitude_error <= TAKEOFF_ALTITUDE_TOLERANCE

        if altitude_ready:
            if self.takeoff_stable_start_time is None:
                self.takeoff_stable_start_time = now

            if now - self.takeoff_stable_start_time >= TAKEOFF_STABLE_DURATION:
                self.takeoff_stable = True
        else:
            self.takeoff_stable_start_time = None

    def update_avoidance_enable_state(self, now: float):
        """
        이륙 안정화 이후에만 장애물 회피 판단을 활성화한다.
        """
        self.update_takeoff_stable_state(now)
        self.avoidance_enabled = self.takeoff_stable

    def reset_velocity_limiter(self):
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_vz = 0.0

    def choose_initial_avoid_sign(self) -> float:
        min_heading = math.radians(MIN_HEADING_FOR_SIDE_DECISION_DEG)

        if abs(self.vfh_selected_heading) >= min_heading:
            return 1.0 if self.vfh_selected_heading > 0.0 else -1.0

        if abs(self.last_nonzero_vfh_heading) >= min_heading:
            return 1.0 if self.last_nonzero_vfh_heading > 0.0 else -1.0

        if self.left_avg > self.right_avg:
            return 1.0

        return -1.0

    def enter_avoid_mode(self, now: float):
        self.mode = "AVOID"
        self.avoid_start_time = now
        _, _, self.avoid_start_s = self.get_projection_point()
        self.return_stable_start_time = None

        self.front_clear_count = 0
        self.avoid_elapsed_for_log = 0.0
        self.avoid_progress_for_log = 0.0

        # AVOID 진입 시 VFH+가 선택한 좌/우 회피 방향을 저장
        self.avoid_sign = self.choose_initial_avoid_sign()

        # AVOID 진입 직후 heading이 너무 작으면 초기 회피 방향을 조금 부여
        min_heading = math.radians(MIN_HEADING_FOR_SIDE_DECISION_DEG)
        if abs(self.smoothed_vfh_heading) < min_heading:
            self.smoothed_vfh_heading = self.avoid_sign * min_heading

        self.reset_velocity_limiter()

    def update_mode(self):
        now = time.time()
        _, _, current_s = self.get_projection_point()

        self.update_avoidance_enable_state(now)

        enter_avoid = (
            self.avoidance_enabled
            and self.front_min <= VFH_ENTER_AVOID_DISTANCE
        )

        if self.mode == "FOLLOW_PATH":
            self.smoothed_vfh_heading = 0.0
            self.prev_smoothed_vfh_heading = 0.0
            self.front_clear_count = 0
            self.avoid_sign = 0.0


            if enter_avoid:
                self.enter_avoid_mode(now)

        elif self.mode == "AVOID":
            avoid_elapsed = now - getattr(self, "avoid_start_time", now)
            avoid_progress = current_s - getattr(self, "avoid_start_s", current_s)

            self.avoid_elapsed_for_log = avoid_elapsed
            self.avoid_progress_for_log = avoid_progress

            front_is_clear = (
                self.front_min >= CLEAR_FRONT_DISTANCE
                and self.vfh_planner_state == "CRUISE"
            )

            side_is_clear = self.side_min >= CLEAR_SIDE_DISTANCE

            clear_environment = front_is_clear and side_is_clear

            if clear_environment:
                self.front_clear_count += 1
            else:
                self.front_clear_count = 0

            clear_to_return = (
                avoid_elapsed >= MIN_AVOID_TIME
                and clear_environment
                and self.front_clear_count >= CLEAR_COUNT_THRESHOLD
            )

            if clear_to_return:
                self.mode = "RETURN_PATH"
                self.return_stable_start_time = None
                self.smoothed_vfh_heading = 0.0
                self.prev_smoothed_vfh_heading = 0.0
                self.front_clear_count = 0
                self.reset_velocity_limiter()

        elif self.mode == "RETURN_PATH":
            self.front_clear_count = 0

            reenter_avoid = (
                self.avoidance_enabled
                and (
                    self.front_min <= RETURN_REENTER_DISTANCE
                    or self.side_min <= RETURN_REENTER_SIDE_DISTANCE
                    or (
                        self.vfh_planner_state != "CRUISE"
                        and self.front_min <= VFH_ENTER_AVOID_DISTANCE
                    )
                )
            )

            if reenter_avoid:
                self.enter_avoid_mode(now)
                return

            path_error = abs(self.get_path_error())

            if path_error < PATH_TOLERANCE:
                if self.return_stable_start_time is None:
                    self.return_stable_start_time = now
                elif now - self.return_stable_start_time >= 1.0:
                    self.mode = "FOLLOW_PATH"
                    self.return_stable_start_time = None
                    self.smoothed_vfh_heading = 0.0
                    self.prev_smoothed_vfh_heading = 0.0
                    self.avoid_sign = 0.0
                    self.reset_velocity_limiter()
            else:
                self.return_stable_start_time = None

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

        # velocity setpoint 제어
        offboard.position = False
        offboard.velocity = True
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

    def get_stable_vfh_heading(self):
        """
        VFH+가 계산한 raw heading을 저역통과 필터로 부드럽게 만든다.
        회피 방향을 고정하지 않고, VFH+ 방향을 매 주기 반영한다.
        """
        max_steer = math.radians(self.vfh_config.max_steer_deg)
        min_heading = math.radians(MIN_HEADING_FOR_SIDE_DECISION_DEG)

        raw_heading = max(
        -max_steer,
        min(max_steer, self.vfh_selected_heading)
        )

        # AVOID 중에는 좌/우 회피 부호가 순간적으로 바뀌지 않도록 유지한다.
        # 각도 크기는 VFH+ 결과를 사용하고, 부호만 AVOID 진입 시 선택한 방향으로 맞춘다.
        if self.mode == "AVOID" and self.avoid_sign != 0.0:
            if abs(raw_heading) >= min_heading:
                raw_heading = self.avoid_sign * abs(raw_heading)

        # AVOID 중 VFH+가 순간적으로 CRUISE를 반환하더라도,
        # RETURN_PATH로 넘어가기 전까지는 이전 유효 회피 방향을 잠깐 유지한다.
        if self.mode == "AVOID" and self.vfh_planner_state == "CRUISE":
            if self.front_clear_count < CLEAR_COUNT_THRESHOLD:
                if abs(self.last_nonzero_vfh_heading) >= min_heading:
                    raw_heading = self.last_nonzero_vfh_heading
                elif abs(self.smoothed_vfh_heading) >= min_heading:
                    raw_heading = self.smoothed_vfh_heading

        # raw_heading이 너무 작으면 좌우 공간 중 더 넓은 쪽으로 최소 회피각 부여
        if abs(raw_heading) < min_heading and self.front_min <= VFH_ENTER_AVOID_DISTANCE:
            if self.mode == "AVOID" and self.avoid_sign != 0.0:
                raw_heading = self.avoid_sign * min_heading
            elif self.left_avg > self.right_avg:
                raw_heading = min_heading
            else:
                raw_heading = -min_heading

        # 장애물이 가까울수록 새 VFH+ 방향을 빠르게 반영
        if self.front_min <= 1.5:
            alpha = 0.75
        elif self.front_min <= 3.0:
            alpha = 0.60
        elif self.front_min <= 5.0:
            alpha = 0.45
        else:
            alpha = 0.25

        stable_heading = (
            (1.0 - alpha) * self.smoothed_vfh_heading
            + alpha * raw_heading
        )

        stable_heading = max(-max_steer, min(max_steer, stable_heading))

        # 너무 가까운 경우 최소 회피각 보장
        if self.front_min <= 1.2:
            min_abs_heading = math.radians(75.0)
        elif self.front_min <= 2.0:
            min_abs_heading = math.radians(65.0)
        elif self.front_min <= 3.0:
            min_abs_heading = math.radians(50.0)
        elif self.front_min <= 5.0:
            min_abs_heading = math.radians(30.0)
        else:
            min_abs_heading = 0.0

        if 0.0 < abs(stable_heading) < min_abs_heading:
            sign = 1.0 if stable_heading > 0.0 else -1.0
            stable_heading = sign * min_abs_heading

        stable_heading = max(-max_steer, min(max_steer, stable_heading))

        self.prev_smoothed_vfh_heading = self.smoothed_vfh_heading
        self.smoothed_vfh_heading = stable_heading

        return stable_heading

    def make_follow_target(self):
        target_x = self.x + self.path_dir_x * FOLLOW_TARGET_DISTANCE
        target_y = self.y + self.path_dir_y * FOLLOW_TARGET_DISTANCE
        return target_x, target_y

    def make_avoid_target(self):
        stable_heading = self.get_stable_vfh_heading()

        if self.front_min <= 1.2:
            target_distance = AVOID_TARGET_DISTANCE
            forward_scale = 0.00
        elif self.front_min <= 2.0:
            target_distance = AVOID_TARGET_DISTANCE
            forward_scale = 0.10
        elif self.front_min <= 3.0:
            target_distance = AVOID_TARGET_DISTANCE
            forward_scale = 0.25
        elif self.front_min <= 5.0:
            target_distance = AVOID_TARGET_DISTANCE
            forward_scale = 0.50
        else:
            target_distance = AVOID_TARGET_DISTANCE
            forward_scale = 0.80

        c = math.cos(stable_heading)
        s = math.sin(stable_heading)

        dir_x = self.path_dir_x * max(0.0, c) * forward_scale + self.path_left_x * s
        dir_y = self.path_dir_y * max(0.0, c) * forward_scale + self.path_left_y * s

        norm = math.sqrt(dir_x * dir_x + dir_y * dir_y)

        if norm < 1e-6:
            dir_x = self.path_dir_x
            dir_y = self.path_dir_y
        else:
            dir_x /= norm
            dir_y /= norm

        target_x = self.x + dir_x * target_distance
        target_y = self.y + dir_y * target_distance

        return target_x, target_y, stable_heading

    def make_return_target(self, path_error: float):
        return_gain = 0.18

        dir_x = self.path_dir_x - self.path_left_x * path_error * return_gain
        dir_y = self.path_dir_y - self.path_left_y * path_error * return_gain

        norm = math.sqrt(dir_x * dir_x + dir_y * dir_y)

        if norm < 1e-6:
            dir_x = self.path_dir_x
            dir_y = self.path_dir_y
        else:
            dir_x /= norm
            dir_y /= norm

        target_x = self.x + dir_x * RETURN_TARGET_DISTANCE
        target_y = self.y + dir_y * RETURN_TARGET_DISTANCE

        return target_x, target_y

    def limit_velocity_step(self, vx: float, vy: float, vz: float):
        dvx = vx - self.prev_vx
        dvy = vy - self.prev_vy
        dvz = vz - self.prev_vz

        dxy = math.sqrt(dvx * dvx + dvy * dvy)

        if dxy > MAX_VEL_STEP:
            scale = MAX_VEL_STEP / dxy
            vx = self.prev_vx + dvx * scale
            vy = self.prev_vy + dvy * scale

        if abs(dvz) > MAX_VEL_STEP:
            vz = self.prev_vz + math.copysign(MAX_VEL_STEP, dvz)

        self.prev_vx = vx
        self.prev_vy = vy
        self.prev_vz = vz

        return vx, vy, vz

    def make_velocity_command(self, target_x: float, target_y: float):
        dx = target_x - self.x
        dy = target_y - self.y

        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            dir_x = 0.0
            dir_y = 0.0
        else:
            dir_x = dx / dist
            dir_y = dy / dist

        if self.mode == "FOLLOW_PATH":
            speed = FOLLOW_SPEED
        elif self.mode == "AVOID":
            speed = AVOID_SPEED
        elif self.mode == "RETURN_PATH":
            speed = RETURN_SPEED
        else:
            speed = FOLLOW_SPEED

        vx = dir_x * speed
        vy = dir_y * speed

        xy_speed = math.sqrt(vx * vx + vy * vy)

        if xy_speed > MAX_XY_SPEED:
            scale = MAX_XY_SPEED / xy_speed
            vx *= scale
            vy *= scale

        target_z = self.z if self.target_z is None else self.target_z

        z_error = target_z - self.z
        vz = ALTITUDE_GAIN * z_error

        if vz > MAX_Z_SPEED:
            vz = MAX_Z_SPEED
        elif vz < -MAX_Z_SPEED:
            vz = -MAX_Z_SPEED

        # 이륙 안정화 전에는 수평 이동하지 않는다.
        if not self.takeoff_stable:
            vx = 0.0
            vy = 0.0

        vx, vy, vz = self.limit_velocity_step(vx, vy, vz)

        return vx, vy, vz

    def publish_velocity_setpoint(self, timestamp: int, vx: float, vy: float, vz: float):
        sp = TrajectorySetpoint()
        sp.timestamp = timestamp

        sp.position = [
            float("nan"),
            float("nan"),
            float("nan")
        ]

        sp.velocity = [
            float(vx),
            float(vy),
            float(vz)
        ]

        sp.acceleration = [
            float("nan"),
            float("nan"),
            float("nan")
        ]

        if self.path_initialized:
            sp.yaw = float(math.atan2(self.path_dir_y, self.path_dir_x))
        else:
            sp.yaw = 0.0

        self.traj_pub.publish(sp)

    def control(self, timestamp):
        if not self.path_initialized:
            z_error = 0.0 if self.target_z is None else self.target_z - self.z
            vz = ALTITUDE_GAIN * z_error

            if vz > MAX_Z_SPEED:
                vz = MAX_Z_SPEED
            elif vz < -MAX_Z_SPEED:
                vz = -MAX_Z_SPEED

            vx, vy, vz = self.limit_velocity_step(0.0, 0.0, vz)
            self.publish_velocity_setpoint(timestamp, vx, vy, vz)

            print(
                f"mode=WAIT_POSITION | "
                f"z={self.z:.2f} | "
                f"target_z={self.target_z} | "
                f"takeoff_stable={self.takeoff_stable} | "
                f"avoid_enabled={self.avoidance_enabled} | "
                f"vel=({vx:.2f}, {vy:.2f}, {vz:.2f})"
            )
            return

        self.update_mode()

        _, _, s = self.get_projection_point()
        path_error = self.get_path_error()

        stable_heading_for_log = self.smoothed_vfh_heading

        if self.mode == "FOLLOW_PATH":
            target_x, target_y = self.make_follow_target()

        elif self.mode == "AVOID":
            target_x, target_y, stable_heading_for_log = self.make_avoid_target()

        elif self.mode == "RETURN_PATH":
            target_x, target_y = self.make_return_target(path_error)

        else:
            target_x, target_y = self.make_follow_target()

        vx, vy, vz = self.make_velocity_command(target_x, target_y)

        self.publish_velocity_setpoint(timestamp, vx, vy, vz)

        if stable_heading_for_log > math.radians(5.0):
            current_side = "LEFT"
        elif stable_heading_for_log < -math.radians(5.0):
            current_side = "RIGHT"
        else:
            current_side = "STRAIGHT"

        altitude_error = 0.0 if self.target_z is None else abs(self.target_z - self.z)

        print(
            f"mode={self.mode} | "
            f"front={self.front_min:.2f} | "
            f"left_min={self.left_min:.2f} | "
            f"right_min={self.right_min:.2f} | "
            f"side_min={self.side_min:.2f} | "
            f"path_error={path_error:.2f} | "
            f"vfh_state={self.vfh_planner_state} | "
            f"raw_heading={math.degrees(self.vfh_selected_heading):.1f} deg | "
            f"smooth_heading={math.degrees(stable_heading_for_log):.1f} deg | "
            f"side={current_side} | "
            f"takeoff_stable={self.takeoff_stable} | "
            f"avoid_enabled={self.avoidance_enabled} | "
            f"v_close={self.v_close:.2f} | "
            f"alt_err={altitude_error:.2f} | "
            f"avoid_t={self.avoid_elapsed_for_log:.1f} | "
            f"avoid_prog={self.avoid_progress_for_log:.1f} | "
            f"clear_cnt={self.front_clear_count} | "
            f"target=({target_x:.2f}, {target_y:.2f}) | "
            f"vel=({vx:.2f}, {vy:.2f}, {vz:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = VFHPlusDroneControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()