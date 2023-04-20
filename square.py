"""
 airsim 四旋翼飞正方形
"""
import airsim
import numpy as np
import math
import time
import cv2
from yolov5 import yolo_detector
from stereo import stereo_detector
import matplotlib.pyplot as plt

object_detector = yolo_detector.Yolo()
distance_detector = stereo_detector.StereoDetector()

client = airsim.MultirotorClient()  # connect to the AirSim simulator
client.reset()
client.enableApiControl(True, vehicle_name='Drone1')       # 获取控制权
client.armDisarm(True, vehicle_name='Drone1')              # 解锁
client.takeoffAsync(vehicle_name='Drone1').join()        # 起飞
client.moveToZAsync(-3, 1, vehicle_name='Drone1').join()   # 第二阶段：上升到2米高度

client.enableApiControl(True, vehicle_name='Drone0')       # 获取控制权
client.armDisarm(True, vehicle_name='Drone0')              # 解锁
client.takeoffAsync(vehicle_name='Drone0').join()        # 起飞
client.moveToZAsync(-5, 1, vehicle_name='Drone0').join()   # 第二阶段：上升到2米高度

# 轨迹参数
tra_time = 200                      # 200*0.2

# 环境参数
pos_drone0 = np.array([5., 0., -4.])   # 实际上高度应该是跟上面 -3 保持一致，但是检测存在一个固定误差，可能是因为相机位置导致

# 记录数据
x_real = []
y_real = []
z_real = []
x_det = []
y_det = []
z_det = []
value = 0   # 记录有多少条有用数据
real = []
det = []

# 速度控制
for i in range(tra_time):
    # 控制目标飞正方形
    if i < tra_time/8:
        client.moveByVelocityAsync(0, 1, 0, duration=1, vehicle_name='Drone1')
    elif i < tra_time/8*3:
        client.moveByVelocityAsync(-1, 0, 0, duration=1, vehicle_name='Drone1')
    elif i < tra_time/8*5:
        client.moveByVelocityAsync(0, -1, 0, duration=1, vehicle_name='Drone1')
    elif i < tra_time/8*7:
        client.moveByVelocityAsync(1, 0, 0, duration=1, vehicle_name='Drone1')
    else:
        client.moveByVelocityAsync(0, 1, 0, duration=1, vehicle_name='Drone1')
    start = time.time()
    # 第二部分：Drone0测位置
    if i % 3 == 0:
        # 1.拍照
        # count = i
        # print(f"1.开始拍照，count：{count}")
        responses = client.simGetImages([
            airsim.ImageRequest('cam_left', airsim.ImageType.Scene, pixels_as_float=False, compress=False),
            airsim.ImageRequest('cam_right', airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        ], 'Drone0')
        img_left = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height,
                                                                                        responses[0].width, 3)
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        img_right = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(responses[1].height,
                                                                                         responses[1].width, 3)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        # 2.目标检测
        # print(f"2.开始目标检测，count：{count}")
        xyxy_l, conf_l, cls_l = object_detector.run_detection(img_left)
        xyxy_r, conf_r, cls_r = object_detector.run_detection(img_right)
        has_object = False
        if xyxy_l and xyxy_r:
            has_object = True
        else:
            print(f"\t没有检测到目标")

        # 检测到目标，则测距
        if has_object:
            # 3.双目测距
            # print(f"3.开始测距，count：{count}")
            x = int((xyxy_l[0] + xyxy_l[2] + xyxy_r[0] + xyxy_r[2]) / 4)
            y = int((xyxy_l[1] + xyxy_l[3] + xyxy_r[1] + xyxy_r[3]) / 4)
            distance, pos_cam = distance_detector.run_detect(img_left, img_right, x, y)
            # print(f"\t目标在左相机坐标系下的坐标：{pos_cam}")

            # 验证双目测距的有效性
            finite = False
            valid = True
            if math.isfinite(pos_cam[0]) and math.isfinite(pos_cam[1]) and math.isfinite(pos_cam[2]):
                finite = True
            else:
                print(f"\t双目测距数据无效")
            # if (np.sum(pre_position) == 0) or np.sqrt(np.sum((pos_cam - pre_position) ** 2)) < 10:
            #     valid = True
            #     pre_position = pos_cam  # 这个地方可能有bug，如果第一次执行，if的第一个条件成立，如果此时pos_cam是一个非常离谱的值，就会有问题
            # else:
            #     print(f"\t双目测距数据无效")

            # 4.如果测距数据有效
            if finite and valid:
                # Drone1坐标转换：左相机坐标系 -> 机体坐标系 -> 世界坐标系
                R_cam2body = np.array(
                    [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])  # pos_cam=[XYZ]，这个坐标XYZ是在相机坐标系下，而机体坐标系下pos_body=[ZXY]
                T_cam2body = np.array([0.15, -0.2, 0.3])
                pos_body = np.dot(R_cam2body, pos_cam.T) + T_cam2body.T
                # print(f"\t目标在机体坐标系下被检测到的坐标：{pos_body}")  # 此时如果Drone0没有动（姿态没有变化），那么在机体坐标系下的坐标，就是在世界坐标系下的坐标
                R_body2NED = np.array(
                    [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])  # 这个机体坐标系到世界坐标系的旋转坐标是根据Drone0的偏航角yaw决定的，当前yaw=180
                pos_NED = np.dot(R_body2NED, pos_body)  # R*Body=NED，当然此时只是坐标系的轴重合，但是原点不重合（因为Drone0已经起飞）
                pos_NED = pos_NED + pos_drone0  # Drone1位置测量值，在世界坐标系下
                pos_real = client.getMultirotorState('Drone1').kinematics_estimated.position
                pos_real = np.array([pos_real.x_val, pos_real.y_val, pos_real.z_val])

                # 对数据进一步判断，目前只是一个不严谨的判断
                if pos_NED[2] < 0 and np.linalg.norm(pos_NED, ord=2) < 50:
                    print(pos_NED)
                    print(pos_real)
                    # 保存数据，方便画图
                    x_det.append(pos_NED[0])
                    y_det.append(pos_NED[1])
                    z_det.append(pos_NED[2])
                    x_real.append(pos_real[0])
                    y_real.append(pos_real[1])
                    z_real.append(pos_real[2])
                    print('\n')

                    # 展示检测点在图片上的位置
                    # cv2.circle(img_left, (int(x), int(y)), radius=2, color=[0, 0, 255])
                    # cv2.imshow("img_left", img_left)
                    # cv2.waitKey()
                    # cv2.circle(img_right, (int(x), int(y)), radius=2, color=[0, 0, 255])
                    # cv2.imshow("img_right", img_right)
                    # cv2.waitKey()
                    # 保存图片，生成数据集
                    strl = "image/square/left/" + str(value) + ".png"
                    strr = "image/square/right/" + str(value) + ".png"
                    iml = cv2.imwrite(strl, img_left)
                    imr = cv2.imwrite(strr, img_right)
                    real.append(pos_real.tolist())
                    det.append(pos_NED.tolist())

                    value = value + 1
                else:
                    print('最终数据无效，双目测距点不在目标点上')
    else:
        time.sleep(0.2)

Note = open('image/square/real_pos.txt', mode='w')
Note.write(str(real))
Note = open('image/square/det_pos.txt', mode='w')
Note.write(str(det))
Note.close()
print(f"有效记录数据为 {value}，占比 {value / (tra_time / 5)}")
index = range(1, len(x_real) + 1)
plt.plot(index, x_real, 'ob:', label='The true value')
plt.plot(index, x_det, 'or:', label='The measured value')
plt.legend()
plt.title('The x axis')
plt.show()
plt.plot(index, y_real, 'ob:', label='The true value')
plt.plot(index, y_det, 'or:', label='The measured value')
plt.legend()
plt.title('The y axis')
plt.show()
plt.plot(index, z_real, 'ob:', label='The true value')
plt.plot(index, z_det, 'or:', label='The measured value')
plt.legend()
plt.title('The z axis')
plt.show()

