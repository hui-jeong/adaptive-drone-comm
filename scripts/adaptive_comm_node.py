#상태에 따라 주기/QoS/depth 변경
import json
import math
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleAttitude,
    SensorCombined,
)


RAW_SCAN_TOPIC = "/world/default/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan"
RAW_POSITION_TOPIC = "/fmu/out/vehicle_local_position_v1"
RAW_ATTITUDE_TOPIC = "/fmu/out/vehicle_attitude"
RAW_SENSOR_COMBINED_TOPIC = "/fmu/out/sensor_combined"

EXP_SCAN_TOPIC = "/experiment/critical/scan"
EXP_POSITION_TOPIC = "/experiment/critical/local_position"
EXP_ATTITUDE_TOPIC = "/experiment/critical/attitude"
EXP_SENSOR_COMBINED_TOPIC = "/experiment/critical/sensor_combined"

EXP_NONCRITICAL_TOPIC = "/experiment/noncritical/load"
EXP_POLICY_TOPIC = "/experiment/comm_policy"
LEGACY_ADAPTIVE_POLICY_TOPIC = "/experiment/adaptive_policy"


CASE_CONFIG = {
    # Case 1: 중속 + 낮은 통신 부하
    1: {"normal_critical_hz": 50.0, "normal_noncritical_hz": 5.0, "payload_bytes": 1024},

    # Case 2: 중속 + 중간 통신 부하
    2: {"normal_critical_hz": 50.0, "normal_noncritical_hz": 20.0, "payload_bytes": 8 * 1024},

    # Case 3: 중속 + 높은 통신 부하
    3: {"normal_critical_hz": 50.0, "normal_noncritical_hz": 50.0, "payload_bytes": 32 * 1024},

    # Case 4: 고속 + 높은 통신 부하
    4: {"normal_critical_hz": 50.0, "normal_noncritical_hz": 50.0, "payload_bytes": 32 * 1024},
}


@dataclass(frozen=True)
class CommPolicy:
    state: str
    critical_hz: float
    noncritical_hz: float
    reliability: ReliabilityPolicy
    depth: int

    @property
    def reliability_name(self):
        if self.reliability == ReliabilityPolicy.RELIABLE:
            return "RELIABLE"
        return "BEST_EFFORT"


class AdaptiveCommNode(Node):
    """
    Adaptive 통신 노드.

    원본 PX4/Gazebo 토픽을 받아 실험용 토픽으로 재발행한다.
    장애물 접근 거리와 접근 속도(v_close)에 따라 critical 주기, non-critical 주기,
    reliability, depth 정책을 바꾼다.
    """

    def __init__(self):
        super().__init__("adaptive_comm_node")

        self.declare_parameter("case_id", 1)
        self.case_id = int(self.get_parameter("case_id").value)
        self.config = CASE_CONFIG.get(self.case_id, CASE_CONFIG[1])

        self.normal_critical_hz = float(self.config["normal_critical_hz"])
        self.normal_noncritical_hz = float(self.config["normal_noncritical_hz"])
        self.payload_bytes = int(self.config["payload_bytes"])

        self.latest_pos = None
        self.latest_scan = None
        self.latest_attitude = None
        self.latest_sensor_combined = None

        self.front_min = 10.0
        self.prev_front_min = None
        self.prev_scan_time = None
        self.v_close = 0.0

        self.state = "NORMAL"
        self.state_enter_time = time.time()

        self.last_critical_pub_time = 0.0
        self.last_noncritical_pub_time = 0.0
        self.last_policy_pub_time = 0.0

        self.seq = 0
        self.payload = "x" * self.payload_bytes

        self.current_policy = self.make_policy("NORMAL")
        self.current_qos_key = None

        px4_sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pos_sub = self.create_subscription(
            VehicleLocalPosition,
            RAW_POSITION_TOPIC,
            self.pos_callback,
            px4_sub_qos,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            RAW_SCAN_TOPIC,
            self.scan_callback,
            10,
        )

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            RAW_ATTITUDE_TOPIC,
            self.attitude_callback,
            px4_sub_qos,
        )

        self.sensor_combined_sub = self.create_subscription(
            SensorCombined,
            RAW_SENSOR_COMBINED_TOPIC,
            self.sensor_combined_callback,
            px4_sub_qos,
        )

        self.policy_pub = self.create_publisher(String, EXP_POLICY_TOPIC, 10)
        self.legacy_policy_pub = self.create_publisher(String, LEGACY_ADAPTIVE_POLICY_TOPIC, 10)

        self.pos_pub = None
        self.scan_pub = None
        self.attitude_pub = None
        self.sensor_combined_pub = None
        self.noncritical_pub = None

        self.recreate_publishers_if_needed(self.current_policy)

        self.timer = self.create_timer(0.005, self.timer_callback)

        self.get_logger().info(
            f"Adaptive comm started | case={self.case_id} | "
            f"normal_critical={self.normal_critical_hz}Hz | "
            f"normal_noncritical={self.normal_noncritical_hz}Hz | "
            f"payload={self.payload_bytes}B"
        )

    def make_qos(self, reliability: ReliabilityPolicy, depth: int) -> QoSProfile:
        return QoSProfile(
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
        )

    def make_policy(self, state: str) -> CommPolicy:
        if state == "NORMAL":
            return CommPolicy(
                state="NORMAL",
                critical_hz=self.normal_critical_hz,
                noncritical_hz=self.normal_noncritical_hz,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=10,
            )

        if state == "CAUTION":
            return CommPolicy(
                state="CAUTION",
                critical_hz=max(50.0, self.normal_critical_hz),
                noncritical_hz=max(2.0, self.normal_noncritical_hz * 0.4),
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=5,
            )

        if state in ("DANGER", "AVOID"):
            return CommPolicy(
                state=state,
                critical_hz=max(50.0, self.normal_critical_hz),
                noncritical_hz=1.0,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=3,
            )

        if state == "RETURN":
            return CommPolicy(
                state="RETURN",
                critical_hz=max(50.0, self.normal_critical_hz * 0.75),
                noncritical_hz=2.0,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=5,
            )

        return self.make_policy("NORMAL")

    def recreate_publishers_if_needed(self, policy: CommPolicy):
        qos_key = (policy.reliability_name, policy.depth)

        if qos_key == self.current_qos_key:
            return

        if self.pos_pub is not None:
            self.destroy_publisher(self.pos_pub)

        if self.scan_pub is not None:
            self.destroy_publisher(self.scan_pub)

        if self.attitude_pub is not None:
            self.destroy_publisher(self.attitude_pub)       

        if self.sensor_combined_pub is not None:
            self.destroy_publisher(self.sensor_combined_pub)

        if self.noncritical_pub is not None:
            self.destroy_publisher(self.noncritical_pub)

        qos = self.make_qos(policy.reliability, policy.depth)

        self.pos_pub = self.create_publisher(
            VehicleLocalPosition,
            EXP_POSITION_TOPIC,
            qos,
        )

        self.scan_pub = self.create_publisher(
            LaserScan,
            EXP_SCAN_TOPIC,
            qos,
        )

        self.attitude_pub = self.create_publisher(
            VehicleAttitude,
            EXP_ATTITUDE_TOPIC,
            qos,
        )

        self.sensor_combined_pub = self.create_publisher(
            SensorCombined,
            EXP_SENSOR_COMBINED_TOPIC,
            qos,
        )

        self.noncritical_pub = self.create_publisher(
            String,
            EXP_NONCRITICAL_TOPIC,
            qos,
        )

        self.current_qos_key = qos_key

        self.get_logger().info(
            f"QoS changed | state={policy.state} | "
            f"reliability={policy.reliability_name} | "
            f"depth={policy.depth}"
        )

    def pos_callback(self, msg):
        self.latest_pos = msg

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.front_min = self.compute_front_min(msg)
        self.update_v_close()

    def attitude_callback(self, msg):
        self.latest_attitude = msg

    def sensor_combined_callback(self, msg):
        self.latest_sensor_combined = msg

    def compute_front_min(self, msg: LaserScan) -> float:
        front_half_angle = math.radians(30.0)
        min_dist = 10.0
        angle = msg.angle_min

        for r in msg.ranges:
            norm_angle = self.normalize_angle(angle)

            if -front_half_angle <= norm_angle <= front_half_angle:
                if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                    min_dist = min(min_dist, r)

            angle += msg.angle_increment

        return min_dist

    def update_v_close(self):
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

            # LiDAR 튐 방지
            if raw_v_close > 5.0:
                raw_v_close = 0.0

            alpha = 0.7
            self.v_close = alpha * self.v_close + (1.0 - alpha) * raw_v_close

        self.prev_front_min = self.front_min
        self.prev_scan_time = now

    def update_state(self):
        now = time.time()

        # =========================
        # Dynamic distance thresholds
        # =========================
        # DANGER는 충돌 직전이 아니라,
        # 회피 제어가 본격적으로 필요한 통신 긴급 상태로 정의한다.
        d_caution_enter = max(6.5, self.v_close * 3.0 + 1.0)
        d_danger_enter = max(4.5, self.v_close * 2.0 + 1.0)

        # hysteresis
        # 진입 기준보다 빠져나오는 기준을 더 크게 둬서 상태가 튀는 것을 막는다.
        d_danger_exit = d_danger_enter + 0.5
        d_caution_exit = d_caution_enter + 0.5

        old_state = self.state
        new_state = old_state

        if old_state == "NORMAL":
            if self.front_min <= d_danger_enter:
                new_state = "DANGER"
            elif self.front_min <= d_caution_enter:
                new_state = "CAUTION"
            else:
                new_state = "NORMAL"

        elif old_state == "CAUTION":
            if self.front_min <= d_danger_enter:
                new_state = "DANGER"
            elif self.front_min >= d_caution_exit:
                new_state = "RETURN"
            else:
                new_state = "CAUTION"

        elif old_state == "DANGER":
            if self.front_min >= d_caution_exit:
                new_state = "RETURN"
            elif self.front_min >= d_danger_exit:
                new_state = "CAUTION"
            else:
                new_state = "DANGER"

        elif old_state == "RETURN":
            # 복귀 중 다시 가까워지면 재진입
            if self.front_min <= d_danger_enter:
                new_state = "DANGER"
            elif self.front_min <= d_caution_enter:
                new_state = "CAUTION"
            elif now - self.state_enter_time >= 3.0:
                new_state = "NORMAL"
            else:
                new_state = "RETURN"

        else:
            new_state = "NORMAL"

        if new_state != old_state:
            self.state = new_state
            self.state_enter_time = now
            self.current_policy = self.make_policy(self.state)
            self.recreate_publishers_if_needed(self.current_policy)

            self.get_logger().info(
                f"State changed | {old_state} -> {new_state} | "
                f"front={self.front_min:.2f} | "
                f"v_close={self.v_close:.2f} | "
                f"d_caution_enter={d_caution_enter:.2f} | "
                f"d_danger_enter={d_danger_enter:.2f} | "
                f"critical={self.current_policy.critical_hz}Hz | "
                f"noncritical={self.current_policy.noncritical_hz}Hz | "
                f"qos={self.current_policy.reliability_name} | "
                f"depth={self.current_policy.depth}"
            )

    def timer_callback(self):
        self.update_state()

        now = time.time()
        policy = self.current_policy

        critical_period = 1.0 / policy.critical_hz
        noncritical_period = 1.0 / policy.noncritical_hz

        if now - self.last_critical_pub_time >= critical_period:
            if self.latest_pos is not None:
                self.pos_pub.publish(self.latest_pos)

            if self.latest_scan is not None:
                self.scan_pub.publish(self.latest_scan)

            if self.latest_attitude is not None:
                self.attitude_pub.publish(self.latest_attitude)

            if self.latest_sensor_combined is not None:
                self.sensor_combined_pub.publish(self.latest_sensor_combined)

            self.last_critical_pub_time = now

        if now - self.last_noncritical_pub_time >= noncritical_period:
            msg = String()
            msg.data = (
                f"adaptive|case={self.case_id}|"
                f"state={policy.state}|"
                f"critical_hz={policy.critical_hz}|"
                f"noncritical_hz={policy.noncritical_hz}|"
                f"qos={policy.reliability_name}|"
                f"depth={policy.depth}|"
                f"seq={self.seq}|"
                + self.payload
            )

            self.noncritical_pub.publish(msg)

            self.seq += 1
            self.last_noncritical_pub_time = now

        if now - self.last_policy_pub_time >= 1.0:
            msg = String()
            msg.data = json.dumps(
                {
                    "mode": "adaptive",
                    "state": policy.state,
                    "case_id": self.case_id,
                    "front_min": self.front_min,
                    "v_close": self.v_close,
                    "critical_hz": policy.critical_hz,
                    "noncritical_hz": policy.noncritical_hz,
                    "qos_reliability": policy.reliability_name,
                    "depth": policy.depth,
                    "payload_bytes": self.payload_bytes,
                },
                ensure_ascii=False,
            )

            self.policy_pub.publish(msg)
            self.legacy_policy_pub.publish(msg)
            self.last_policy_pub_time = now

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveCommNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()