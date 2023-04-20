import time
import math
"""
每一个控制量都需要一个PID控制器
"""
class PIDcontroller:
    def __init__(self, kp, ki, kd, max_out=20):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out

        self.error = 0
        self.pre_error = 0
        self.integral = 0

    def getOutPut(self, curr_error):
        self.pre_error = self.error
        self.error = curr_error
        self.integral += curr_error

        P = self.kp * self.error
        I = self.ki * self.integral
        D = self.kd * (self.error - self.pre_error)

        output = P + I + D

        if output > self.max_out:
            output = self.max_out
        if output < -1 * self.max_out:
            output = -1 * self.max_out

        return output



