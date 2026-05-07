#CSV 저장
import csv
import json
import math
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from px4_msgs.msg import VehicleAttitude, SensorCombined
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


CONTROL_STATE_TOPIC = "/experiment/control_state"
COMM_POLICY_TOPIC = "/experiment/comm_policy"
COLLISION_EVENT_TOPIC = "/experiment/collision_event"

ATTITUDE_TOPIC = "/experiment/critical/attitude"
SENSOR_COMBINED_TOPIC = "/experiment/critical/sensor_combined"

    
def to_float_safe(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

class ExperimentLogger(Node):
    """
    VFH+ 제어 노드의 상태 토픽과 통신 정책 토픽을 구독해서 CSV로 저장하는 노드.

    - /experiment/control_state : 회피 제어 상태
    - /experiment/comm_policy   : baseline/adaptive 통신 정책
    - /experiment/collision_event : 충돌 이벤트
    """

    def __init__(self):
        super().__init__("experiment_logger")

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.declare_parameter("experiment_mode", "baseline")
        self.declare_parameter("case_id", 1)
        self.declare_parameter("run_id", 1)
        self.declare_parameter("log_dir", "/home/huijeong/adaptive-drone-comm/logs")

        self.experiment_mode = str(self.get_parameter("experiment_mode").value)
        self.case_id = int(self.get_parameter("case_id").value)
        self.run_id = int(self.get_parameter("run_id").value)
        self.log_dir = str(self.get_parameter("log_dir").value)
        self.start_wall_time = None

        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = Path(self.log_dir) / (
            f"{self.experiment_mode}_case{self.case_id}_run{self.run_id}.csv"
        )

        self.latest_policy = {
            "comm_mode": "",
            "comm_state": "",
            "critical_hz": "",
            "noncritical_hz": "",
            "qos_reliability": "",
            "depth": "",
            "payload_bytes": "",
        }

        self.latest_attitude = {
            "roll_deg": "",
            "pitch_deg": "",
            "yaw_deg": "",
        }

        self.latest_angular_velocity = {
            "roll_rate": "",
            "pitch_rate": "",
            "yaw_rate": "",
        }

        self.actual_collision = 0
        self.collision_time = ""
        self.contact_count = 0
        self.contact_pair = ""

        self.fieldnames = [
            "wall_time",
            "elapsed_time",
            "ros_time_us",
            "experiment_mode",
            "case_id",
            "run_id",

            "mode",
            "x",
            "y",
            "z",

            "front_min",
            "left_min",
            "right_min",
            "side_min",

            "path_error",
            "v_close",
            "vfh_state",
            "raw_heading",
            "smooth_heading",

            "vx",
            "vy",
            "vz",

            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "roll_rate",
            "pitch_rate",
            "yaw_rate",

            "target_x",
            "target_y",

            "takeoff_stable",
            "avoid_enabled",
            "avoid_t",
            "avoid_prog",
            "clear_cnt",

            "comm_mode",
            "comm_state",
            "critical_hz",
            "noncritical_hz",
            "qos_reliability",
            "depth",
            "payload_bytes",

            "actual_collision",
            "collision_time",
            "contact_count",
            "contact_pair",
        ]
        

        self.log_file = open(self.log_path, "w", newline="")
        self.writer = csv.DictWriter(self.log_file, fieldnames=self.fieldnames)
        self.writer.writeheader()

        self.row_count = 0

        self.control_state_sub = self.create_subscription(
            String,
            CONTROL_STATE_TOPIC,
            self.control_state_callback,
            10,
        )

        self.comm_policy_sub = self.create_subscription(
            String,
            COMM_POLICY_TOPIC,
            self.comm_policy_callback,
            10,
        )

        self.collision_sub = self.create_subscription(
            String,
            COLLISION_EVENT_TOPIC,
            self.collision_event_callback,
            10,
        )

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            ATTITUDE_TOPIC,
            self.attitude_callback,
            px4_qos,
        )

        self.sensor_combined_sub = self.create_subscription(
            SensorCombined,
            SENSOR_COMBINED_TOPIC,
            self.sensor_combined_callback,
            px4_qos,
        )

        self.get_logger().info(f"Experiment logger started: {self.log_path}")

    def collision_event_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse collision_event JSON: {e}")
            return

        # 실제 물리 접촉이 발생했을 때만 1
        # 한 번 충돌이 발생하면 해당 실험은 actual_collision=1로 유지
        self.actual_collision = 1
        self.collision_time = data.get("wall_time", "")
        self.contact_count = data.get("contact_count", 0)

        pairs = data.get("contact_pairs", [])

        if pairs:
            first_pair = pairs[0]
            self.contact_pair = (
                f"{first_pair.get('collision1', '')} | "
                f"{first_pair.get('collision2', '')}"
            )
        else:
            self.contact_pair = ""

        self.get_logger().warn(
            f"Actual collision logged | contact_count={self.contact_count}"
        )

    def comm_policy_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        self.latest_policy = {
            "comm_mode": data.get("mode", ""),
            "comm_state": data.get("state", ""),
            "critical_hz": data.get("critical_hz", ""),
            "noncritical_hz": data.get("noncritical_hz", ""),
            "qos_reliability": data.get("qos_reliability", ""),
            "depth": data.get("depth", ""),
            "payload_bytes": data.get("payload_bytes", ""),
        }

    def control_state_callback(self, msg: String):
        try:
            state = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse control_state JSON: {e}")
            return
        
        wall_time = to_float_safe(state.get("wall_time", ""))

        if self.start_wall_time is None:
            self.start_wall_time = wall_time

        elapsed_time = wall_time - self.start_wall_time

        row = {
            "wall_time": state.get("wall_time", ""),
            "elapsed_time": elapsed_time,
            "ros_time_us": state.get("ros_time_us", ""),
            "experiment_mode": self.experiment_mode,
            "case_id": self.case_id,
            "run_id": self.run_id,

            "mode": state.get("mode", ""),
            "x": state.get("x", ""),
            "y": state.get("y", ""),
            "z": state.get("z", ""),

            "front_min": state.get("front_min", ""),
            "left_min": state.get("left_min", ""),
            "right_min": state.get("right_min", ""),
            "side_min": state.get("side_min", ""),

            "path_error": state.get("path_error", ""),
            "v_close": state.get("v_close", ""),
            "vfh_state": state.get("vfh_state", ""),
            "raw_heading": state.get("raw_heading", ""),
            "smooth_heading": state.get("smooth_heading", ""),

            "vx": state.get("vx", ""),
            "vy": state.get("vy", ""),
            "vz": state.get("vz", ""),

            "target_x": state.get("target_x", ""),
            "target_y": state.get("target_y", ""),

            "takeoff_stable": state.get("takeoff_stable", ""),
            "avoid_enabled": state.get("avoid_enabled", ""),
            "avoid_t": state.get("avoid_t", ""),
            "avoid_prog": state.get("avoid_prog", ""),
            "clear_cnt": state.get("clear_cnt", ""),
        }
        
        row.update(self.latest_attitude)
        row.update(self.latest_angular_velocity)
        row.update(self.latest_policy)

        row.update(
            {
                "actual_collision": self.actual_collision,
                "collision_time": self.collision_time,
                "contact_count": self.contact_count,
                "contact_pair": self.contact_pair,
            }
        )

        self.writer.writerow(row)

        self.row_count += 1

        if self.row_count % 20 == 0:
            self.log_file.flush()

    def destroy_node(self):
        if hasattr(self, "log_file"):
            self.log_file.flush()
            self.log_file.close()

        super().destroy_node()

    def quaternion_to_euler_deg(self, q):
        # PX4 VehicleAttitude q는 일반적으로 [w, x, y, z]
        w = float(q[0])
        x = float(q[1])
        y = float(q[2])
        z = float(q[3])

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw),
        )
    
    def attitude_callback(self, msg: VehicleAttitude):
        try:
            roll, pitch, yaw = self.quaternion_to_euler_deg(msg.q)

            self.latest_attitude = {
                "roll_deg": roll,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
            }

        except Exception as e:
            self.get_logger().warn(f"Failed to parse attitude: {e}")


    def sensor_combined_callback(self, msg: SensorCombined):
        try:
            gyro = list(msg.gyro_rad)

            self.latest_angular_velocity = {
                "roll_rate": float(gyro[0]),
                "pitch_rate": float(gyro[1]),
                "yaw_rate": float(gyro[2]),
            }

        except Exception as e:
            self.get_logger().warn(f"Failed to parse sensor_combined: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()