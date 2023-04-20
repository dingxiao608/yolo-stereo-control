"""
飞正方形（速度控制）
"""
import airsim
import time

client = airsim.MultirotorClient()  # connect to the AirSim simulator
client.enableApiControl(True, vehicle_name='Drone1')  # 获取控制权
client.armDisarm(True, vehicle_name='Drone1')  # 解锁
client.takeoffAsync(vehicle_name='Drone1').join()  # 第一阶段：起飞
client.moveToZAsync(-2, 1, vehicle_name='Drone1').join()  # 第二阶段：上升到2米高度
# 飞正方形
# client.moveByVelocityZAsync(1, 0, -2, 8, vehicle_name='Drone1').join()  # 第三阶段：以1m/s速度向前飞8秒钟
# client.moveByVelocityZAsync(0, 1, -2, 8, vehicle_name='Drone1').join()  # 第三阶段：以1m/s速度向右飞8秒钟
# client.moveByVelocityZAsync(-1, 0, -2, 8, vehicle_name='Drone1').join()  # 第三阶段：以1m/s速度向后飞8秒钟
# client.moveByVelocityZAsync(0, -1, -2, 8, vehicle_name='Drone1').join()  # 第三阶段：以1m/s速度向左飞8秒钟
client.moveToPositionAsync(5, 0, -2, velocity=1, vehicle_name='Drone1').join()
client.moveToPositionAsync(5, 5, -2, velocity=1, vehicle_name='Drone1').join()
client.moveToPositionAsync(0, 5, -2, velocity=1, vehicle_name='Drone1').join()
client.moveToPositionAsync(0, 0, -2, velocity=1, vehicle_name='Drone1').join()
# 悬停 2 秒钟
client.hoverAsync(vehicle_name='Drone1').join()  # 第四阶段：悬停6秒钟
time.sleep(6)
client.landAsync(vehicle_name='Drone1').join()  # 第五阶段：降落
client.armDisarm(False, vehicle_name='Drone1')  # 上锁
client.enableApiControl(False, vehicle_name='Drone1')  # 释放控制权