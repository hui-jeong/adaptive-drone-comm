#고정 주기/QoS/depth로 데이터 재발행
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from std_msgs import msg
from std_msgs.msg import String
from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleAttitude,
    SensorCombined,
)

RAW_SCAN_TOPIC = "/world/default/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan"
RAW_POSITION_TOPIC = "/fmu/out/vehicle_local_position_v1"

EXP_SCAN_TOPIC = "/experiment/critical/scan"
EXP_POSITION_TOPIC = "/experiment/critical/local_position"
EXP_NONCRITICAL_TOPIC = "/experiment/noncritical/load"
EXP_POLICY_TOPIC = "/experiment/comm_policy"

RAW_ATTITUDE_TOPIC = "/fmu/out/vehicle_attitude"
RAW_SENSOR_COMBINED_TOPIC = "/fmu/out/sensor_combined"

EXP_ATTITUDE_TOPIC = "/experiment/critical/attitude"
EXP_SENSOR_COMBINED_TOPIC = "/experiment/critical/sensor_combined"


CASE_CONFIG = {
    # Case 1: 중속 + 낮은 통신 부하
    1: {"critical_hz": 50.0, "noncritical_hz": 5.0, "payload_bytes": 1024},

    # Case 2: 중속 + 중간 통신 부하
    2: {"critical_hz": 50.0, "noncritical_hz": 20.0, "payload_bytes": 8 * 1024},

    # Case 3: 중속 + 높은 통신 부하
    3: {"critical_hz": 50.0, "noncritical_hz": 50.0, "payload_bytes": 32 * 1024},

    # Case 4: 고속 + 높은 통신 부하
    4: {"critical_hz": 50.0, "noncritical_hz": 50.0, "payload_bytes": 32 * 1024},
}


class BaselineCommNode(Node):
    """
    Baseline 통신 노드.

    원본 PX4/Gazebo 토픽을 받아서 실험용 토픽으로 재발행한다.
    Baseline에서는 상태와 관계없이 critical/non-critical 주기, QoS, depth를 고정한다.
    """

    def __init__(self):
        super().__init__("baseline_comm_node")

        self.declare_parameter("case_id", 1)
        self.case_id = int(self.get_parameter("case_id").value)
        self.config = CASE_CONFIG.get(self.case_id, CASE_CONFIG[1])

        self.critical_hz = float(self.config["critical_hz"])
        self.noncritical_hz = float(self.config["noncritical_hz"])
        self.payload_bytes = int(self.config["payload_bytes"])
        self.latest_attitude = None
        self.latest_sensor_combined = None

        self.critical_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.noncritical_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        px4_sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.latest_pos = None
        self.latest_scan = None

        self.last_critical_pub_time = 0.0
        self.last_noncritical_pub_time = 0.0
        self.last_policy_pub_time = 0.0

        self.seq = 0
        self.payload = "x" * self.payload_bytes

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

        self.pos_pub = self.create_publisher(
            VehicleLocalPosition,
            EXP_POSITION_TOPIC,
            self.critical_qos,
        )

        self.scan_pub = self.create_publisher(
            LaserScan,
            EXP_SCAN_TOPIC,
            self.critical_qos,
        )

        self.noncritical_pub = self.create_publisher(
            String,
            EXP_NONCRITICAL_TOPIC,
            self.noncritical_qos,
        )

        self.attitude_pub = self.create_publisher(
            VehicleAttitude,
            EXP_ATTITUDE_TOPIC,
            self.critical_qos,
        )

        self.sensor_combined_pub = self.create_publisher(
            SensorCombined,
            EXP_SENSOR_COMBINED_TOPIC,
            self.critical_qos,
        )

        self.policy_pub = self.create_publisher(
            String,
            EXP_POLICY_TOPIC,
            10,
        )
        

        self.timer = self.create_timer(0.005, self.timer_callback)

        self.get_logger().info(
            f"Baseline comm started | case={self.case_id} | "
            f"critical={self.critical_hz}Hz | "
            f"noncritical={self.noncritical_hz}Hz | "
            f"qos=BEST_EFFORT | depth=10 | "
            f"payload={self.payload_bytes}B"
        )

    def pos_callback(self, msg):
        self.latest_pos = msg

    def scan_callback(self, msg):
        self.latest_scan = msg

    def timer_callback(self):
        now = time.time()

        critical_period = 1.0 / self.critical_hz
        noncritical_period = 1.0 / self.noncritical_hz

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
            msg.data = f"baseline|case={self.case_id}|seq={self.seq}|" + self.payload
            self.noncritical_pub.publish(msg)

            self.seq += 1
            self.last_noncritical_pub_time = now

        if now - self.last_policy_pub_time >= 1.0:
            policy_msg = String()
            policy_msg.data = json.dumps(
                {
                    "mode": "baseline",
                    "state": "FIXED",
                    "case_id": self.case_id,
                    "critical_hz": self.critical_hz,
                    "noncritical_hz": self.noncritical_hz,
                    "qos_reliability": "BEST_EFFORT",
                    "depth": 10,
                    "payload_bytes": self.payload_bytes,
                },
                ensure_ascii=False,
            )

            self.policy_pub.publish(policy_msg)
            self.last_policy_pub_time = now

    def attitude_callback(self, msg):
        self.latest_attitude = msg


    def sensor_combined_callback(self, msg):
        self.latest_sensor_combined = msg


def main(args=None):
    rclpy.init(args=args)
    node = BaselineCommNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()