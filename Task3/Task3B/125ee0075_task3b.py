#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math

WAYPOINTS =[[5.5, 9.5],
            [7.0, 5.5],
            [10.0, 5.5],
            [7.8, 3.0],
            [9.0, 0.5],
            [5.5, 2.5],
            [2.0, 0.5],
            [3.2, 3.0],
            [1.0, 5.5],
            [4.0, 5.5],
            [5.5, 9.5]]

DIST_THRESHOLD = 0.15
LINEAR_SPEED = 4.0
ANGULAR_KP = 6.0
HEADING_THRESHOLD = 0.4

class WaypointNavigator(Node):

    def __init__(self):
        super().__init__("waypoint_navigator")
        self.publisher = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.subscriber = self.create_subscription(Pose, "/turtle1/pose", self.pose_callback, 10)
        
        self.waypoint_index = 0
        self.pose = None
        self.done = False

        self.timer = self.create_timer(0.05, self.timer_callback)

    def pose_callback(self, msg):
        self.pose = msg

    def timer_callback(self):
        if self.pose is None or self.done:
            return
        
        if self.waypoint_index >= len(WAYPOINTS):
            self.stop()
            self.done = True
            return
        
        target_x, target_y = WAYPOINTS[self.waypoint_index]
        dx = target_x - self.pose.x
        dy = target_y - self.pose.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < DIST_THRESHOLD:
            self.waypoint_index += 1
            return
        
        desired_theta = math.atan2(dy, dx)
        angle_error = desired_theta - self.pose.theta

        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        cmd = Twist()
        cmd.angular.z = ANGULAR_KP * angle_error

        if abs(angle_error) < HEADING_THRESHOLD:
            cmd.linear.x = LINEAR_SPEED
        else:
            cmd.linear.x = 0.0

        self.publisher.publish(cmd)

    def stop(self):
        self.publisher.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':  
    main()        