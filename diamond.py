import airsim
import numpy as np
import math
import time
import cv2
from yolov5 import yolo_detector
from stereo import stereo_detector
import matplotlib.pyplot as plt


def traj():
    period = 24     # 周期
    seg = math.sqrt(2)
    v_yz = 0.2
    v_x = 0.5
    T1 = period / 4
    T2 = period / 2
    T3 = 3 * period / 4






client = airsim.MultirotorClient()  # connect to the AirSim simulator
client.reset()
client.enableApiControl(True, vehicle_name='Drone1')       # 获取控制权
client.armDisarm(True, vehicle_name='Drone1')              # 解锁
client.takeoffAsync(vehicle_name='Drone1').join()        # 起飞
client.moveToZAsync(-3, 1, vehicle_name='Drone1').join()   # 第二阶段：上升到2米高度

client.enableApiControl(True, vehicle_name='Drone0')       # 获取控制权
client.armDisarm(True, vehicle_name='Drone0')              # 解锁
client.takeoffAsync(vehicle_name='Drone0').join()        # 起飞
client.moveToZAsync(-3, 1, vehicle_name='Drone0').join()   # 第二阶段：上升到2米高度