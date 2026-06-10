#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_interfaces.srv import PrimeFactors
import sys

class PrimeFactorsClient(Node):

    def __init__(self):
        super().__init__("prime_factor_client")
        self.client = self.create_client(PrimeFactors, "prime_factors")

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")


    def send_request(self):
        request = PrimeFactors.Request()
        request.num = int(sys.argv[1]) 
        self.future = self.client.call_async(request)

def main(args=None):
    rclpy.init(args=args)
    prime_factors_client = PrimeFactorsClient()
    prime_factors_client.send_request()

    while rclpy.ok():
        rclpy.spin_once(prime_factors_client)
        if prime_factors_client.future.done():
            try:
                response = prime_factors_client.future.result()
            except Exception as e:
                prime_factors_client.get_logger().info(f"Service call failed: {e}")
            else:
                prime_factors_client.get_logger().info(f"Prime factors: {response.prime_factors}")
            break

    prime_factors_client.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()