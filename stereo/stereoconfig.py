import numpy as np


# 双目相机参数
class StereoCamera(object):
    def __init__(self):
        # 左相机内参（估计值）
        self.cam_matrix_left = np.array([[320., 0., 320.],
                                         [0., 320., 240.],
                                         [0., 0., 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[320., 0., 320.],
                                         [0., 320., 240.],
                                         [0., 0., 1.]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]（真实值）
        self.distortion_l = np.array([[0., 0., 0., 0., 0.]])
        self.distortion_r = np.array([[0., 0., 0., 0., 0.]])

        # 旋转矩阵
        self.R = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]])

        # 平移矩阵（估计值，单位mm）
        self.T = np.array([[-400.], [0.], [0.]])    

        # 焦距fx≈fy≈f（估计值）
        self.focal_length = 320.

        # 基线距离。单位：mm， 为平移向量的第一个参数（取绝对值）（估计值）
        self.baseline = 400

