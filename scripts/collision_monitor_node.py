#실제 물리 접촉 발생 여부
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ros_gz_interfaces.msg import Contacts


CONTACT_TOPIC = "/experiment/gz/obstacle_contacts"
COLLISION_EVENT_TOPIC = "/experiment/collision_event"


class CollisionMonitorNode(Node):
    """
    Gazebo contact sensor 정보를 받아 실제 물리 충돌 여부를 판단하는 노드.

    contact가 하나라도 감지되면 /experiment/collision_event로
    actual_collision=1 이벤트를 발행한다.
    """

    def __init__(self):
        super().__init__("collision_monitor_node")

        self.declare_parameter("contact_topic", CONTACT_TOPIC)
        self.contact_topic = str(self.get_parameter("contact_topic").value)

        self.collision_pub = self.create_publisher(
            String,
            COLLISION_EVENT_TOPIC,
            10,
        )

        self.contact_sub = self.create_subscription(
            Contacts,
            self.contact_topic,
            self.contact_callback,
            10,
        )

        self.collision_detected = False

        self.get_logger().info(
            f"Collision monitor started | contact_topic={self.contact_topic}"
        )

    def contact_callback(self, msg: Contacts):
        contact_count = len(msg.contacts)

        # 실제 접촉이 없으면 아무 이벤트도 보내지 않음
        if contact_count <= 0:
            return

        self.collision_detected = True

        contact_pairs = []

        for contact in msg.contacts:
            collision1 = getattr(contact, "collision1", "")
            collision2 = getattr(contact, "collision2", "")

            contact_pairs.append(
                {
                    "collision1": collision1,
                    "collision2": collision2,
                }
            )

        event = {
            "wall_time": time.time(),
            "actual_collision": 1,
            "contact_count": contact_count,
            "contact_pairs": contact_pairs,
        }

        out = String()
        out.data = json.dumps(event, ensure_ascii=False)
        self.collision_pub.publish(out)

        self.get_logger().warn(
            f"ACTUAL COLLISION DETECTED | contact_count={contact_count}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CollisionMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()