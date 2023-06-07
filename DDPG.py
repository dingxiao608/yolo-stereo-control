import json
import math
import time

import airsim
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D

from stereo import stereo_detector
from yolov5 import yolo_detector


filename = "C:/Users/Administrator/Documents/AirSim/settings.json"
with open(filename) as f:
    settings = json.load(f)
init_pos0 = [settings["Vehicles"]["Drone0"]["X"], settings["Vehicles"]["Drone0"]["Y"],
             settings["Vehicles"]["Drone0"]["Z"]]
init_pos1 = [settings["Vehicles"]["Drone1"]["X"], settings["Vehicles"]["Drone1"]["Y"],
             settings["Vehicles"]["Drone1"]["Z"]]


class AirsimEnv:
    def __init__(self, init_pos=np.array([init_pos0, init_pos1])):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.init_pos = init_pos

        self.start_time = None
        self.drone0_trajx = []
        self.drone0_trajy = []
        self.drone0_trajz = []
        self.drone1_trajx = []
        self.drone1_trajy = []
        self.drone1_trajz = []

    def __del__(self):
        print("exiting")
        time.sleep(1)
        self.client.armDisarm(False, vehicle_name="Drone0")
        self.client.armDisarm(False, vehicle_name="Drone1")
        self.client.enableApiControl(False, vehicle_name="Drone0")
        self.client.enableApiControl(False, vehicle_name="Drone1")

    def init_drone(self, vehicle_name):
        self.client.enableApiControl(True, vehicle_name=vehicle_name)
        self.client.armDisarm(True, vehicle_name=vehicle_name)
        self.client.armDisarm(False, vehicle_name=vehicle_name)
        self.client.armDisarm(True, vehicle_name=vehicle_name)

    def reset_env(self):
        """
        重置无人机，drone0、drone1都飞到z-10m
        """
        self.client.reset()
        time.sleep(0.2)

        self.init_drone(vehicle_name="Drone0")
        self.init_drone(vehicle_name="Drone1")

        self.client.takeoffAsync(vehicle_name="Drone0").join()
        time.sleep(0.2)
        self.client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Drone0").join()  # 这个方法是以每个无人机自身初始位置为原点

        self.client.takeoffAsync(vehicle_name="Drone1").join()
        time.sleep(0.2)
        self.client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Drone1").join()

        self.client.hoverAsync(vehicle_name="Drone0").join()
        self.client.hoverAsync(vehicle_name="Drone1").join()

        self.start_time = time.time() - 10      # 为什么 -10s
        self.drone0_trajx = []
        self.drone0_trajy = []
        self.drone0_trajz = []
        self.drone1_trajx = []
        self.drone1_trajy = []
        self.drone1_trajz = []

    def get_observation(self):
        """
        用drone0上的真实传感器获取观测图像。奖励用真实数据计算。done
        :return: observation、reward、done
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest('cam_left', airsim.ImageType.Scene, pixels_as_float=False, compress=False),
            airsim.ImageRequest('cam_right', airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        ], 'Drone0')
        img_left = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height,
                                                                                        responses[0].width, 3)
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        img_right = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(responses[1].height,
                                                                                         responses[1].width, 3)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        # 真实位置计算reward
        pos_drone0 = self.client.getMultirotorState('Drone0').kinematics_estimated.position
        pos_drone0 = np.array([pos_drone0.x_val, pos_drone0.y_val, pos_drone0.z_val]) + self.init_pos[0]
        pos_drone1 = self.client.getMultirotorState('Drone1').kinematics_estimated.position
        pos_drone1 = np.array([pos_drone1.x_val, pos_drone1.y_val, pos_drone1.z_val]) + self.init_pos[1]
        vec = pos_drone1 - pos_drone0

        # 距离奖励
        horizontal_distance = np.linalg.norm(vec[:-1])
        if abs(horizontal_distance - 7) < 3:         # 水平跟踪距离在 7±3
            reward = 5
        else:
            reward = -5

        vertical_distance = vec[-1]
        if abs(vertical_distance) < 3:              # 垂直跟踪距离在 0±3
            reward += 5
        else:
            reward -= 5
        # 偏航角奖励
        yaw_relative = math.degrees(math.atan2(vec[1], vec[0]))
        pitch_radian, roll_radian, yaw_radian = airsim.to_eularian_angles(self.client.getMultirotorState('Drone0')
                                                                          .kinematics_estimated.orientation)
        yaw_drone0 = math.degrees(yaw_radian)   # yaw_drone0角度制
        yaw_diff = yaw_drone0 - yaw_relative
        if abs(yaw_diff) > 180:
            yaw_diff = 360 - yaw_diff
        if abs(yaw_diff) < 25:
            reward += 5
        else:
            reward -= 5

        # 如果目标丢失，是否应该done？
        done = horizontal_distance > 16 or horizontal_distance < 4 or vertical_distance > 4

        return img_left, img_right, yaw_radian, pos_drone0, reward, done

    def step(self, vx=0, vy=0, vz=0, yaw=0):
        """
        传入动作action，vx、vy、vz是NED下面的速度，yaw是偏航角（单位°）
        """
        self.client.moveByVelocityAsync(vx, vy, vz, duration=1, yaw_mode=airsim.YawMode(False, yaw),
                                        vehicle_name='Drone0')

        # Drone1随机移动
        if time.time() - self.start_time > 10:
            x = np.random.randint(-200, 200)
            y = np.random.randint(-200, 200)
            z = np.random.randint(25, 75)
            self.client.moveToPositionAsync(x, y, -z, 3, vehicle_name="Drone1")     # 目标无人机速度为3m/s，不指定速度默认是1m/s
            print("drone1飞行")
            self.start_time = time.time()


class ViewDrone:
    def __init__(self):
        self.object_detector = yolo_detector.Yolo()
        self.distance_detector = stereo_detector.StereoDetector()

    # 输入左右相机图片，输出检测结果
    # 问题：检测不到目标应该返回什么？检测到目标但是测距点无效应该返回什么？目前是None
    def run(self, img_left, img_right, yaw_radian, pos_drone0):
        xyxy_l, conf_l, cls_l = self.object_detector.run_detection(img_left)
        xyxy_r, conf_r, cls_r = self.object_detector.run_detection(img_right)
        if xyxy_l and xyxy_r:
            # 问题，这个点（x,y）不一定在目标上
            x = int((xyxy_l[0] + xyxy_l[2] + xyxy_r[0] + xyxy_r[2]) / 4)
            y = int((xyxy_l[1] + xyxy_l[3] + xyxy_r[1] + xyxy_r[3]) / 4)
            distance, pos_cam = self.distance_detector.run_detect(img_left, img_right, x, y)
            if math.isfinite(pos_cam[0]) and math.isfinite(pos_cam[1]) and math.isfinite(pos_cam[2]):
                # 第一次坐标转换：左相机坐标系 -> 机体坐标系
                R_cam2body = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
                T_cam2body = np.array([0.15, -0.2, 0.3])
                pos_body = np.dot(R_cam2body, pos_cam.T) + T_cam2body.T

                # 第二次坐标转换：机体坐标系 -> 世界坐标系（以Drone0起始位置为世界坐标系原点）
                R_body2NED = np.array([
                    [math.cos(yaw_radian), -math.sin(yaw_radian), 0],
                    [math.sin(yaw_radian), math.cos(yaw_radian), 0],
                    [0, 0, 1]
                ])
                pos_NED = np.dot(R_body2NED, pos_body)  # R*Body=NED，当然此时只是坐标系的轴重合，但是原点不重合（因为Drone0已经起飞）
                pos_det = pos_drone0 + pos_NED  # Drone1在世界坐标系下的检测位置
                if pos_det[2] < 0 and np.linalg.norm(pos_det, ord=2) < 50:
                    return pos_det


class ReplayBuffer:
    def __init__(self, mem_size):
        self.storage = []
        self.max_size = mem_size
        self.mem_cnt = 0

    # state, next_state, action, reward, np.float(done)
    def store_transition(self, transition):
        if len(self.storage) == self.max_size:
            index = self.mem_cnt % self.max_size
            self.storage[index] = transition
            self.mem_cnt += 1
        else:
            self.storage.append(transition)

    def sample_buffer(self, batch_size):
        batch = np.random.randint(0, len(self.storage), size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []

        for i in batch:
            s, ns, a, r, d = self.storage[i]
            state.append(np.array(s, copy=False))   # np.array(s, copy=False) 将list s转化为numpy数组
            next_state.append(np.array(ns, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(next_state), np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, device, directory, mem_size=1000000, batch_size=256,
                 gamma=0.99, tau=0.005):
        self.device = device
        self.directory = directory  # TensorBoard日志文件存储路径
        self.gamma = gamma
        self.tau = tau

        # actor and target_actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        # critic and target_critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(mem_size)
        self.batch_size = batch_size

        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def learn(self):
        if self.replay_buffer.mem_cnt < self.batch_size:
            return

        for it in range(200):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample_buffer(self.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update the target network
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1


if __name__ == '__main__':
    env = AirsimEnv()
    env.reset_env()
    a = 20
    print(f"start time: {time.time()}")
    while a > 0:
        a -= 1
        env.step(1, 0, 0, 0)
        time.sleep(1)
    print("停止飞行")
    print(f"end time: {time.time()}")
    time.sleep(5)
    print("主程序结束")