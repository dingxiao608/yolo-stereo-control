import time

import airsim
import myapi
import datetime
import math
import numpy as np
from yolov5 import yolo_detector
from stereo import stereo_detector
import cv2


object_detector = yolo_detector.Yolo()
distance_detector = stereo_detector.StereoDetector()

client = airsim.MultirotorClient()
client.reset()
client.enableApiControl(True, "Drone0")
client.armDisarm(True, "Drone0")
client.enableApiControl(True, "Drone1")
client.armDisarm(True, "Drone1")


# for i in range(8):
#     i = i + 1
#     i = str(i)
#     strl = "image/left/" + i + ".png"
#     strr = "image/right/" + i + ".png"
#     iml = cv2.imread(strl, cv2.IMREAD_COLOR)
#     imr = cv2.imread(strr, cv2.IMREAD_COLOR)
#     cv2.imshow("iml", iml)
#     cv2.waitKey()
# 目标检测
# xyxyl, confl, clsl = object_detector.run_detection(iml)
# xyxyr, confr, clsr = object_detector.run_detection(imr)
# if xyxyl and xyxyr:
#     cv2.imwrite("image/left/9.png", iml)
#     cv2.imwrite("image/right/9.png", imr)
#     print("save")
    # # 双目测距
    # x = int((xyxyl[0] + xyxyl[2] + xyxyr[0] + xyxyr[2]) / 4)
    # y = int((xyxyl[1] + xyxyl[3] + xyxyr[1] + xyxyr[3]) / 4)
    # distance, position = distance_detector.run_detect(iml1, imr1, x, y)
    # print('点(%d, %d)的坐标是(%f, %f, %f)，距离是%f' % (x, y, position[0], position[1], position[2], distance))
    #
    # # 左相机坐标系 -> 机体坐标系
    # R = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    # T = np.array([0.15, -0.2, 0.3])
    # pos_body = np.dot(R, position.T) + T.T
    # print(pos_body)
    #
    # # Drone1相对于Drone0的真实位置（参考系是NED，原点在Drone0起点）
    # pos1 = myapi.getRealPosition(client=client, name="Drone1")
    # pos0 = myapi.getRealPosition(client=client, name="Drone0")
    # pos_relative = np.array([pos1.x_val - pos0.x_val, pos1.y_val - pos0.y_val, pos1.z_val - pos0.z_val])
    # print(pos_relative)

# # 识别 + 测距
# start_time = time.time()
# xyxyl, confl, clsl = object_detector.run_detection(iml)
# xyxyr, confr, clsr = object_detector.run_detection(imr)
# mid_time = time.time()
# # print(mid_time - start_time)
# x = int((xyxyl[0] + xyxyl[2] + xyxyr[0] + xyxyr[2]) / 4)
# y = int((xyxyl[1] + xyxyl[3] + xyxyr[1] + xyxyr[3]) / 4)
# distance, position = distance_detector.run_detect(iml, imr, x, y)
# print('点(%d, %d)的坐标是(%f, %f, %f)，距离是%f' % (x, y, position[0], position[1], position[2], distance))
# # print(time.time() - mid_time)
# R = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
# T = np.array([0.15, -0.2, 0.3])
# pos_body = np.dot(R, position.T) + T.T
# print(pos_body)
# pos1 = myapi.getRealPosition(client=client, name="Drone1")
# pos0 = myapi.getRealPosition(client=client, name="Drone0")
# pos_relative = np.array([pos1.x_val - pos0.x_val, pos1.y_val - pos0.y_val, pos1.z_val - pos0.z_val])
# print(pos_relative)


myapi.close_drones(client)

