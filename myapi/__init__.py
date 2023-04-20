import time

import airsim
import numpy as np

# 初始位置定义
orig_pos = np.array([[5., 0., 0.],
                     [0., 0., 0.]])  # -1.6是起飞后的高度


# 无人机位置控制函数封装
def move_to_position(client, x, y, z, velocity, name='', waited=True,
                     drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode()):
    num = int(name[-1])
    x -= orig_pos[num, 0]
    y -= orig_pos[num, 1]
    z -= orig_pos[num, 2]
    if waited:
        client.moveToPositionAsync(x, y, z, velocity, drivetrain=drivetrain, yaw_mode=yaw_mode, vehicle_name=name).join()
    else:
        client.moveToPositionAsync(x, y, z, velocity, drivetrain=drivetrain, yaw_mode=yaw_mode, vehicle_name=name)
    return


def getRealPosition(client, name=''):
    num = int(name[-1])
    pos = client.getMultirotorState(name).kinematics_estimated.position
    offset = airsim.Vector3r(orig_pos[num, 0], orig_pos[num, 1], orig_pos[num, 2])
    return pos + offset


def init_drones(client):
    client.enableApiControl(True, "Drone0")
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone0")
    client.armDisarm(True, "Drone1")
    time.sleep(2)


def close_drones(client, drone0="Drone0", drone1="Drone1"):
    client.armDisarm(False, drone0)
    client.armDisarm(False, drone1)
    client.enableApiControl(False, drone0)
    client.enableApiControl(False, drone1)
