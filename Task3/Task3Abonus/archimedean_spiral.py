#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
 
MAX_RADIUS = 5.0
A = 0.05  
B = 0.15  
OMEGA = 0.65 
 
class ArchimedeanSpiral(Node):
 
    def __init__(self):
        super().__init__("archimedean_spiral")
        self.publisher = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.timer = self.create_timer(0.05, self.timer_callback) #Publish per 0.05 sec
 
        self.theta   = 0.0 
        self.stopped = False
 
    def current_radius(self):
        return A + B * self.theta
 
    def stop_turtle(self):
        msg = Twist() 
        self.publisher.publish(msg)
        self.stopped = True
 
    def timer_callback(self):
        if self.stopped is True:
            return
 
        r = self.current_radius()

        if r >= MAX_RADIUS:
            self.stop_turtle()
            return
 
        v = math.sqrt(B**2 + r**2) * OMEGA 
        msg = Twist()
        msg.linear.x  = v
        msg.angular.z = OMEGA
        self.publisher.publish(msg)
 
        self.theta += OMEGA * 0.05
 
 
def main(args=None):
    rclpy.init(args=args)
    node = ArchimedeanSpiral()
    rclpy.spin(node)
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
 