import math
import time

import airsim
import numpy as np
import cv2
import myapi
from PIDcontroller import PIDcontroller
from yolov5 import yolo_detector
from stereo import stereo_detector


"""
相对于linear_noYaw，drone0引入偏航角控制
"""
def linear_Yaw():
    object_detector = yolo_detector.Yolo()
    distance_detector = stereo_detector.StereoDetector()
    client = airsim.MultirotorClient(ip='127.0.0.1')

    client.reset()
    client.enableApiControl(True, "Drone0")
    client.armDisarm(True, "Drone0")
    client.moveToZAsync(-3, 1, vehicle_name='Drone0').join()
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")
    client.moveToZAsync(-3, 1, vehicle_name='Drone1').join()

    # drone1移动
    # client.moveByVelocityAsync(-1, 0, 0, 20, vehicle_name='Drone1')  # Drone1一直保持vx=1向前移动
    # points = [airsim.Vector3r(-10, 0, -3),
    #           airsim.Vector3r(-10, 10, -3),
    #           airsim.Vector3r(0, 10, -3),
    #           airsim.Vector3r(0, 0, -3)]
    points = [airsim.Vector3r(-10, 0, -3),
              airsim.Vector3r(-20, 8, -3),
              airsim.Vector3r(-30, 0, -3),
              airsim.Vector3r(-40, -8, -3)]
    client.moveOnPathAsync(points, 1.5, vehicle_name='Drone1')

    init_pos = np.array([[0, 0, 0],  # -1 是来解决机身高度或者说相机高度问题，有一个固定误差1
                         [-5, 0, 0]])  # 过程中调用 client.getMultirotorState('Drone0').kinematics_estimated.position就必须用到init_pos求真实位置

    while True:
        # 1.拍照
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
            """
            如何确保测距的点在无人机上，仍然是一个问题
            """
            x = int((xyxy_l[0] + xyxy_l[2] + xyxy_r[0] + xyxy_r[2]) / 4)
            y = int((xyxy_l[1] + xyxy_l[3] + xyxy_r[1] + xyxy_r[3]) / 4)
            distance, pos_cam = distance_detector.run_detect(img_left, img_right, x, y)

            # 验证双目测距的有效性
            finite = False
            if math.isfinite(pos_cam[0]) and math.isfinite(pos_cam[1]) and math.isfinite(pos_cam[2]):
                finite = True
            else:
                print(f"\t双目测距数据无效1")

            # 4.如果测距数据有效，则跟踪
            if finite:
                # 第一次坐标转换：左相机坐标系 -> 机体坐标系 -> 世界坐标系
                R_cam2body = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
                T_cam2body = np.array([0.15, -0.2, 0.3])
                pos_body = np.dot(R_cam2body, pos_cam.T) + T_cam2body.T

                # 第二次坐标转换：机体坐标系 -> 世界坐标系（以Drone0起始位置为世界坐标系原点）
                # R_body2NED = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])   # 这个矩阵要根据Drone0的姿态来调整
                yaw = airsim.to_eularian_angles(
                    client.getMultirotorState(vehicle_name='Drone0').kinematics_estimated.orientation)[2]
                R_body2NED = np.array([
                    [math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]
                ])
                pos_NED = np.dot(R_body2NED, pos_body)  # R*Body=NED，当然此时只是坐标系的轴重合，但是原点不重合（因为Drone0已经起飞）
                pos_drone0 = client.getMultirotorState('Drone0').kinematics_estimated.position
                pos_drone0 = np.array([pos_drone0.x_val, pos_drone0.y_val, pos_drone0.z_val]) + init_pos[0]
                pos_det = pos_drone0 + pos_NED  # Drone1在世界坐标系下的检测位置

                if pos_det[2] < 0 and np.linalg.norm(pos_det, ord=2) < 50:
                    # pos_drone1 = client.getMultirotorState('Drone1').kinematics_estimated.position
                    # pos_real = np.array([pos_drone1.x_val, pos_drone1.y_val, pos_drone1.z_val]) + init_pos[1]
                    # print(f'real pos：{pos_real}')
                    # print(f'det pos：{pos_det}')
                    pid_yawcontrol(pos_drone0, pos_det, client)



"""
这个方法需要改进的地方：
1、双目测距不能保证一定测的是无人机上面的点。
3、没有drone0的偏航角控制。
"""
def linear_noYaw():
    object_detector = yolo_detector.Yolo()
    distance_detector = stereo_detector.StereoDetector()
    client = airsim.MultirotorClient(ip='127.0.0.1')

    client.reset()
    client.enableApiControl(True, "Drone0")
    client.armDisarm(True, "Drone0")
    client.moveToZAsync(-3, 1, vehicle_name='Drone0').join()
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")
    client.moveToZAsync(-5, 1, vehicle_name='Drone1').join()

    # drone1移动
    client.moveByVelocityAsync(-1, 0, 0, 20, vehicle_name='Drone1')  # Drone1一直保持vx=1向前移动

    init_pos = np.array([[0, 0, 0],        # -1 是来解决机身高度或者说相机高度问题，有一个固定误差1
                         [-5, 0, 0]])    # 过程中调用 client.getMultirotorState('Drone0').kinematics_estimated.position就必须用到init_pos求真实位置

    while True:
        # 1.拍照
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
            """
            如何确保测距的点在无人机上，仍然是一个问题
            """
            x = int((xyxy_l[0] + xyxy_l[2] + xyxy_r[0] + xyxy_r[2]) / 4)
            y = int((xyxy_l[1] + xyxy_l[3] + xyxy_r[1] + xyxy_r[3]) / 4)
            distance, pos_cam = distance_detector.run_detect(img_left, img_right, x, y)

            # 验证双目测距的有效性
            finite = False
            if math.isfinite(pos_cam[0]) and math.isfinite(pos_cam[1]) and math.isfinite(pos_cam[2]):
                finite = True
            else:
                print(f"\t双目测距数据无效1")


            # 4.如果测距数据有效，则跟踪
            if finite:
                # 第一次坐标转换：左相机坐标系 -> 机体坐标系 -> 世界坐标系
                R_cam2body = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
                T_cam2body = np.array([0.15, -0.2, 0.3])
                pos_body = np.dot(R_cam2body, pos_cam.T) + T_cam2body.T

                # 第二次坐标转换：机体坐标系 -> 世界坐标系（以Drone0起始位置为世界坐标系原点）
                # R_body2NED = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])   # 这个矩阵要根据Drone0的姿态来调整
                yaw = airsim.to_eularian_angles(client.getMultirotorState(vehicle_name='Drone0').kinematics_estimated.orientation)[2]
                R_body2NED = np.array([
                    [math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]
                ])
                pos_NED = np.dot(R_body2NED, pos_body)                              # R*Body=NED，当然此时只是坐标系的轴重合，但是原点不重合（因为Drone0已经起飞）
                pos_drone0 = client.getMultirotorState('Drone0').kinematics_estimated.position
                pos_drone0 = np.array([pos_drone0.x_val, pos_drone0.y_val, pos_drone0.z_val]) + init_pos[0]
                pos_det = pos_drone0 + pos_NED    # Drone1在世界坐标系下的检测位置

                if pos_det[2] < 0 and np.linalg.norm(pos_det, ord=2) < 50:
                    pos_drone1 = client.getMultirotorState('Drone1').kinematics_estimated.position
                    pos_real = np.array([pos_drone1.x_val, pos_drone1.y_val, pos_drone1.z_val]) + init_pos[1]
                    print(f'real pos：{pos_real}')
                    print(f'det pos：{pos_det}')

                    pid_control(pos_drone0, pos_det, client)


def no_Yaw():
    """
    发现目标（目标静止），然后移动到目标位置
    """
    object_detector = yolo_detector.Yolo()
    distance_detector = stereo_detector.StereoDetector()

    client = airsim.MultirotorClient(ip='127.0.0.1')
    client.reset()
    client.enableApiControl(True, "Drone0")
    client.armDisarm(True, "Drone0")

    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")
    client.moveToZAsync(-2, 1, vehicle_name='Drone1').join()

    # 拍照
    responses = client.simGetImages([
        airsim.ImageRequest('cam_left', airsim.ImageType.Scene, pixels_as_float=False, compress=False),
        airsim.ImageRequest('cam_right', airsim.ImageType.Scene, pixels_as_float=False, compress=False)
    ], 'Drone0')
    img_left = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(responses[1].height, responses[1].width, 3)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
    print("images ok....")

    # 目标检测
    xyxy_l, conf_l, cls_l = object_detector.run_detection(img_left)
    xyxy_r, conf_r, cls_r = object_detector.run_detection(img_right)
    print(xyxy_l)
    print(xyxy_r)

    # 检测目标
    if xyxy_l and xyxy_r:
        print("测距........")
        # 双目测距
        x = int((xyxy_l[0] + xyxy_l[2] + xyxy_r[0] + xyxy_r[2]) / 4)
        y = int((xyxy_l[1] + xyxy_l[3] + xyxy_r[1] + xyxy_r[3]) / 4)
        distance, position = distance_detector.run_detect(img_left, img_right, x, y)
        print(position)

        # 左相机坐标系 -> 机体坐标系
        R = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        T = np.array([0.15, -0.2, 0.3])
        pos_body = np.dot(R, position.T) + T.T
        print(f"目标在机体坐标系下被检测到的坐标：{pos_body}")           # 此时因为Drone0没有动，所以在集体坐标系下的坐标，就是在世界坐标系下的坐标

        pos1 = myapi.getRealPosition(client=client, name="Drone1")
        pos0 = myapi.getRealPosition(client=client, name="Drone0")
        pos_relative = np.array([pos1.x_val - pos0.x_val, pos1.y_val - pos0.y_val, pos1.z_val - pos0.z_val])
        print(f"目标在机体坐标系下的真实坐标：{pos_relative}")

        client.moveToZAsync(-0.5, 1, vehicle_name='Drone0').join()   # 必须要先起飞

        # 调用控制方法，飞到指定位置
        pid_control(pos_body, client)

        # pos = client.getMultirotorState('Drone0').kinematics_estimated.position
        # pos = [pos.x_val, pos.y_val, pos.z_val]
        #
        # vx_pid = PIDcontroller(1.2, 0.01, 0.001, 2)
        # vy_pid = PIDcontroller(1.2, 0.01, 0.001, 2)
        # vz_pid = PIDcontroller(1.2, 0.01, 0.001, 2)
        #
        # des = [pos_body[0], pos_body[1], pos_body[2]]
        #
        # x_error = des[0] - pos[0]
        # y_error = des[1] - pos[1]
        # z_error = des[2] - pos[2]
        #
        # while abs(x_error) > 1 or abs(y_error) > 1 or abs(z_error) > 1:
        #     vx_output = vx_pid.getOutPut(x_error)
        #     vy_output = vy_pid.getOutPut(y_error)
        #     vz_output = vz_pid.getOutPut(z_error)
        #
        #     client.moveByVelocityAsync(vx_output, vy_output, vz_output, 0.1, vehicle_name='Drone0').join()  # 这个方法的vz是负值，才是往上飞
        #
        #     pos = client.getMultirotorState('Drone0').kinematics_estimated.position
        #     pos = [pos.x_val, pos.y_val, pos.z_val]
        #     print(pos)
        #     x_error = des[0] - pos[0]
        #     y_error = des[1] - pos[1]
        #     z_error = des[2] - pos[2]


def pid_control(pos, des, client):
    """
    移动Drone0到des，des是一个列表[x, y, z]
    :param pos: drone0位置
    :param des: 目的地
    :param client:
    """
    # pid 控制无人机到 des 位置
    # client = airsim.MultirotorClient()
    # client.reset()
    # client.enableApiControl(True, "Drone0")
    # client.armDisarm(True, "Drone0")
    # client.moveToZAsync(-4, 1, vehicle_name='Drone0').join()
    # print(f"目标位置：{des}")

    vx_pid = PIDcontroller(1.2, 0.01, 0.001, 2)
    vy_pid = PIDcontroller(1.2, 0.01, 0.001, 2)
    vz_pid = PIDcontroller(1.2, 0.01, 0.001, 1)

    x_error = des[0] - pos[0]
    y_error = des[1] - pos[1]
    z_error = des[2] - pos[2]

    while abs(x_error) > 1 or abs(y_error) > 1 or abs(z_error) > 0.5:
        vx_output = vx_pid.getOutPut(x_error)
        vy_output = vy_pid.getOutPut(y_error)
        vz_output = vz_pid.getOutPut(z_error)
        # print(f"pid 输出的vx，vy，vz：{vx_output}，{vy_output}，{vz_output}")
        client.moveByVelocityAsync(vx_output, vy_output, vz_output, 0.1, vehicle_name='Drone0').join()  # 这个方法的vz是负值，才是往上飞

        pos = client.getMultirotorState('Drone0').kinematics_estimated.position
        pos = [pos.x_val, pos.y_val, pos.z_val]
        x_error = des[0] - pos[0]
        y_error = des[1] - pos[1]
        z_error = des[2] - pos[2]


def pid_yawcontrol(pos, des, client):
    # print('跟踪')
    vx_pid = PIDcontroller(1.5, 0.01, 0.001, 2)
    vy_pid = PIDcontroller(1.5, 0.01, 0.001, 2)
    vz_pid = PIDcontroller(1, 0.01, 0.001, 1)

    x_error = des[0] - pos[0]
    y_error = des[1] - pos[1]
    z_error = des[2] - pos[2]

    while abs(x_error) > 3 or abs(y_error) > 3 or abs(z_error) > 0.5:
        vx_output = vx_pid.getOutPut(x_error)
        vy_output = vy_pid.getOutPut(y_error)
        vz_output = vz_pid.getOutPut(z_error)

        angle = math.degrees(math.atan2(y_error, x_error))  # 世界坐标下的偏航角

        client.moveByVelocityAsync(vx_output, vy_output, vz_output, 0.1, yaw_mode=airsim.YawMode(False, angle),
                                   vehicle_name='Drone0').join()  # 这个方法的vz是负值，才是往上飞

        pos = client.getMultirotorState('Drone0').kinematics_estimated.position
        pos = [pos.x_val, pos.y_val, pos.z_val]
        x_error = des[0] - pos[0]
        y_error = des[1] - pos[1]
        z_error = des[2] - pos[2]


def linear_noYaw2():
    object_detector = yolo_detector.Yolo()
    distance_detector = stereo_detector.StereoDetector()
    vx_pid = PIDcontroller(1.2, 0.01, 0.001, 2)
    vy_pid = PIDcontroller(1.2, 0.01, 0.001, 2)
    vz_pid = PIDcontroller(1.2, 0.01, 0.001, 1)

    client = airsim.MultirotorClient(ip='127.0.0.1')
    client.reset()
    client.enableApiControl(True, "Drone0")
    client.armDisarm(True, "Drone0")
    client.takeoffAsync(vehicle_name='Drone0').join()
    client.moveToZAsync(-3, 1, vehicle_name='Drone0').join()
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")
    client.takeoffAsync(vehicle_name='Drone1').join()
    client.moveToZAsync(-3, 1, vehicle_name='Drone1').join()

    # drone1移动
    client.moveByVelocityAsync(-1, 0, 0, 20, vehicle_name='Drone1')  # Drone1一直保持vx=1向前移动

    # 全局初始位置信息
    init_pos = np.array([[0, 0, 0],  # -1 是来解决机身高度或者说相机高度问题，有一个固定误差1
                         [-5, 0, 0]])  # 过程中调用 client.getMultirotorState('Drone0').kinematics_estimated.position就必须用到init_pos求真实位置

    while True:
        # 1.拍照
        responses = client.simGetImages([
            airsim.ImageRequest('cam_left', airsim.ImageType.Scene, pixels_as_float=False, compress=False),
            airsim.ImageRequest('cam_right', airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        ], 'Drone0')
        img_left = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height,
                                                                                        responses[0].width, 3)
        img_right = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(responses[1].height,
                                                                                         responses[1].width, 3)
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        # 2.目标检测
        xyxy_l, conf_l, cls_l = object_detector.run_detection(img_left)
        xyxy_r, conf_r, cls_r = object_detector.run_detection(img_right)
        has_object = False
        if xyxy_l and xyxy_r and (abs(xyxy_l[0] - xyxy_r[0]) < 50):
            has_object = True
        else:
            print(f"\t没有检测到目标")

        # 检测到目标，则测距
        if has_object:
            # 3.双目测距
            x = int((xyxy_l[0] + xyxy_l[2] + xyxy_r[0] + xyxy_r[2]) / 4)
            y = int((xyxy_l[1] + xyxy_l[3] + xyxy_r[1] + xyxy_r[3]) / 4)
            distance, pos_cam = distance_detector.run_detect(img_left, img_right, x, y)

            # 验证双目测距的有效性
            finite = False
            if math.isfinite(pos_cam[0]) and math.isfinite(pos_cam[1]) and math.isfinite(pos_cam[2]):
                finite = True
            else:
                print(f"\t双目测距数据无效1")

            # 4.如果测距数据有效，则跟踪
            if finite:
                # Drone1坐标转换：左相机坐标系 -> 机体坐标系 -> 世界坐标系
                R_cam2body = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
                T_cam2body = np.array([0.15, -0.2, 0.3])
                pos_body = np.dot(R_cam2body, pos_cam.T) + T_cam2body.T

                R_body2NED = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])  # 这个值要根据Drone0的姿态来调整
                pos_NED = np.dot(R_body2NED, pos_body)  # R*Body=NED，当然此时只是坐标系的轴重合，但是原点不重合（因为Drone0已经起飞）
                pos_drone0 = client.getMultirotorState('Drone0').kinematics_estimated.position
                pos_drone0 = np.array([pos_drone0.x_val, pos_drone0.y_val, pos_drone0.z_val]) + init_pos[0]  # 当前位置+初始位置
                pos_det = pos_drone0 + pos_NED  # Drone1在世界坐标系下的检测位置

                # 再次判断测距数据是否有效
                if pos_NED[2] < 0 and np.linalg.norm(pos_NED, ord=2) < 50:
                    pos_drone1 = client.getMultirotorState('Drone1').kinematics_estimated.position
                    pos_real = np.array([pos_drone1.x_val, pos_drone1.y_val, pos_drone1.z_val]) + init_pos[1]   # 当前位置+初始位置
                    print(f'real pos：{pos_real}')
                    print(f'det pos：{pos_det}')

                    x_error = pos_det[0] - pos_drone0[0]
                    y_error = pos_det[1] - pos_drone0[1]
                    z_error = pos_det[2] - pos_drone0[2]
                    if abs(x_error) > 1 or abs(y_error) > 1 or abs(z_error) > 1:
                        vx_output = vx_pid.getOutPut(x_error)
                        vy_output = vy_pid.getOutPut(y_error)
                        vz_output = vz_pid.getOutPut(z_error)
                        # print(f"pid 输出的vx，vy，vz：{vx_output}，{vy_output}，{vz_output}")
                        client.moveByVelocityAsync(vx_output, vy_output, vz_output, 0.1,
                                                   vehicle_name='Drone0')  # 这个方法的vz是负值，才是往上飞


if __name__ == '__main__':
    # pid_control([10, 10, -10], client=airsim.MultirotorClient())
    # no_Yaw()
    # linear_noYaw()
    # linear_noYaw2()
    linear_Yaw()
