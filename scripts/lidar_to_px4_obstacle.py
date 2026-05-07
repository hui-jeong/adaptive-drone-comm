import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from px4_msgs.msg import ObstacleDistance


SCAN_TOPIC = "/world/default/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan"
PX4_TOPIC = "/fmu/in/obstacle_distance"


class LidarToPX4Obstacle(Node):

    def __init__(self):
        super().__init__("lidar_to_px4_obstacle")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self.scan_callback,
            qos
        )

        self.obstacle_pub = self.create_publisher(
            ObstacleDistance,
            PX4_TOPIC,
            10
        )

        self.get_logger().info("LiDAR to PX4 obstacle_distance node started")

    def scan_callback(self, scan_msg):
        obstacle_msg = ObstacleDistance()

        obstacle_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        obstacle_msg.sensor_type = 0
        obstacle_msg.frame = 12

        obstacle_msg.increment = float(math.degrees(scan_msg.angle_increment))
        obstacle_msg.angle_offset = float(math.degrees(scan_msg.angle_min))

        obstacle_msg.min_distance = int(scan_msg.range_min * 100)
        obstacle_msg.max_distance = int(scan_msg.range_max * 100)

        distances = []

        for r in scan_msg.ranges:
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                dist_cm = 65535
            else:
                dist_cm = int(r * 100)

            distances.append(dist_cm)

        # PX4 ObstacleDistance는 72개 거리값 사용
        if len(distances) >= 72:
            distances = distances[:72]
        else:
            distances += [65535] * (72 - len(distances))

        obstacle_msg.distances = distances

        self.obstacle_pub.publish(obstacle_msg)

        front_min = min([d for d in distances if d != 65535], default=65535)

        self.get_logger().info(
            f"Published /fmu/in/obstacle_distance | front_min={front_min / 100:.2f} m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarToPX4Obstacle()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()