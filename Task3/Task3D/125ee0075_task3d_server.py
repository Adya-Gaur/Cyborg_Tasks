#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_interfaces.srv import PrimeFactors

def compute_prime_factors(n):
    factors = []
    if n <= 1:
        return factors
    while n % 2 == 0:
        factors.append(2)
        n //= 2

    divisor = 3
    while divisor**2 <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 2

    if n > 1:
        factors.append(n)

    return factors

class PrimeFactorsServer(Node):
    def __init__(self):
        super().__init__("prime_factors_server")
        self.srv = self.create_service(PrimeFactors, "prime_factors", self.prime_factors_callback)

    def prime_factors_callback(self, request, response):
        n = request.num
        self.get_logger().info(f"Received request for prime factors of: {n}")

        if n < 0:
            response.prime_factors = []
            return response
        
        response.prime_factors = compute_prime_factors(n)
        return response
    
def main(args=None):
    rclpy.init(args=args)
    prime_factors_server = PrimeFactorsServer()
    rclpy.spin(prime_factors_server)
    rclpy.shutdown()

if __name__ == "__main__":
    main()