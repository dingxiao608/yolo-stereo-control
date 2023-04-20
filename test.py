import math
import airsim
import numpy as np


client = airsim.MultirotorClient(ip='127.0.0.1')

client.reset()
client.enableApiControl(True, "Drone0")
client.armDisarm(True, "Drone0")
client.moveToZAsync(-3, 1, vehicle_name='Drone0').join()
client.enableApiControl(True, "Drone1")
client.armDisarm(True, "Drone1")
client.moveToZAsync(-3, 1, vehicle_name='Drone1').join()

init_pos = np.array([[0, 0, 0],        # -1 是来解决机身高度或者说相机高度问题，有一个固定误差1
                         [-5, 0, 0]])

q = client.getMultirotorState(vehicle_name='Drone0').kinematics_estimated.orientation
cur_yaw = airsim.to_eularian_angles(q)[2]
print(f'cur_yaw = {cur_yaw}')

drone1_pos = client.getMultirotorState(vehicle_name='Drone1').kinematics_estimated.position
drone1_pos = np.array([drone1_pos.x_val, drone1_pos.y_val, drone1_pos.z_val]) + init_pos[1]

drone0_pos = client.getMultirotorState(vehicle_name='Drone0').kinematics_estimated.position
drone0_pos = np.array([drone0_pos.x_val, drone0_pos.y_val, drone0_pos.z_val]) + init_pos[0]
vec = [drone1_pos[0] - drone0_pos[0], drone1_pos[1] - drone0_pos[1], drone1_pos[2] - drone0_pos[2]]
print(f'vec = {vec}')
angle = math.degrees(math.atan2(vec[0], vec[1]))    # 输出angle = -90.0， 应该是180or-180
print(f'angle = {angle}')
rela_yaw = 0
if angle > 0 and vec[1] > 0:
    rela_yaw = -(90 - angle)
elif angle > 0 and vec[1] < 0:
    rela_yaw = angle - 90
elif angle < 0 and vec[1] < 0:
    rela_yaw = 270 + angle
elif angle < 0 and vec[1] > 0:
    rela_yaw = -90 + angle
else:
    rela_yaw = 0
print(f'rela_yaw = {rela_yaw}')


