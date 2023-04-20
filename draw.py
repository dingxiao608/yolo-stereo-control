# 画图看真实值和测量值的关联程度

import cv2
from stereo import stereo_detector
from yolov5 import yolo_detector
import numpy as np
import matplotlib.pyplot as plt

def read_txt(fine_name):
    data = []
    file = open(fine_name, 'r')
    file_data = file.readlines()
    for row in file_data:
        tmp_list = row.split(' ')
        tmp_list[-1] = tmp_list[-1].replace('\n', '')
        data.append(tmp_list)
    return data


distance_detector = stereo_detector.StereoDetector()
object_detector = yolo_detector.Yolo()
data = read_txt('image/文件名.txt')
data = np.array(data, dtype=np.int32)
print(data)
x_data = data[:, 0]
y_data = data[:, 1]
z_data = data[:, 2]

x_det = []
y_det = []
z_det = []
# position_data = []
for i in range(1, 9):
    i = str(i)
    strl = "image/left/" + i + ".png"
    strr = "image/right/" + i + ".png"
    iml = cv2.imread(strl, cv2.IMREAD_COLOR)
    imr = cv2.imread(strr, cv2.IMREAD_COLOR)
    xyxy_l, conf_l, cls_l = object_detector.run_detection(iml)
    xyxy_r, conf_r, cls_r = object_detector.run_detection(imr)
    x = int((xyxy_l[0] + xyxy_l[2] + xyxy_r[0] + xyxy_r[2]) / 4)
    y = int((xyxy_l[1] + xyxy_l[3] + xyxy_r[1] + xyxy_r[3]) / 4)
    distance, position = distance_detector.run_detect(iml, imr, x, y)

    # 左相机坐标系 -> 机体坐标系
    R = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    T = np.array([0.15, -0.2, 0.3])
    pos_body = np.dot(R, position.T) + T.T
    # position_data.append(pos_body)
    x_det.append(pos_body[0])
    y_det.append(pos_body[1])
    z_det.append(pos_body[2])

index = range(1, len(x_data) + 1)
print(index)
print(x_data)
print(x_det)
plt.plot(index, x_data, 'ob:', label='The true value')
plt.plot(index, x_det, 'or:', label='The measured value')
plt.legend()
plt.title('The x axis')
plt.show()
plt.plot(index, y_data, 'ob:', label='The true value')
plt.plot(index, y_det, 'or:', label='The measured value')
plt.legend()
plt.title('The y axis')
plt.show()
plt.plot(index, z_data, 'ob:', label='The true value')
plt.plot(index, z_det, 'or:', label='The measured value')
plt.legend()
plt.title('The z axis')
plt.show()
