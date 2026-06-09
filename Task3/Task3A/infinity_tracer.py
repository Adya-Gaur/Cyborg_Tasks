#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

linear_x = 1.0
angular_z = 1.0
circle_time = 2 * math.pi / angular_z
turn_90_time = circle_time / 4

class InfinityTracer(Node):

    def __init__(self):
        super().__init__("infinity_tracer")
        self.pub = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.phase = 0
        self.phase_time = 0.0
        self.timer = self.create_timer(0.05, self.timer_callback)

    def publish_velocity(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def stop(self):
        self.publish_velocity(0.0, 0.0)

    def timer_callback(self):
        if self.phase == 0:
            if self.phase_time < turn_90_time:
                self.publish_velocity(0.0, -angular_z)
                self.phase_time += 0.05
            else:
                self.stop()
                self.phase = 1
                self.phase_time = 0.0

        elif self.phase == 1:
            if self.phase_time < circle_time:
                self.publish_velocity(linear_x, -angular_z)
                self.phase_time += 0.05
            else:
                self.stop()
                self.phase = 2
                self.phase_time = 0.0

        elif self.phase == 2:
            if self.phase_time < circle_time:
                self.publish_velocity(linear_x, angular_z)
                self.phase_time += 0.05
            else:
                self.stop()
                self.phase = 3
                self.timer.cancel()

        else:
            self.stop()

def main(args=None):
    rclpy.init(args=args)
    node = InfinityTracer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
