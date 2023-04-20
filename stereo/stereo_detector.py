import cv2
import numpy
import numpy as np

import stereo.stereo_detect as ss
from stereo import stereoconfig


class StereoDetector():
    def __init__(self):
        self.config = stereoconfig.StereoCamera()

    def run_detect(self, iml, imr, x, y):
        height, width = iml.shape[0:2]

        # 1.消除畸变（airsim原本是没有畸变的）
        iml = ss.undistortion(iml, self.config.cam_matrix_left, self.config.distortion_l)
        imr = ss.undistortion(imr, self.config.cam_matrix_right, self.config.distortion_r)
        # 预处理，一般可以削弱光照不均的影响，不做也可以
        iml_, imr_ = ss.preprocess(iml, imr)

        # 2.立体校正
        # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        map1x, map1y, map2x, map2y, Q = ss.getRectifyTransform(height, width, self.config)
        iml_rectified_l, imr_rectified_r = ss.rectifyImage(iml_, imr_, map1x, map1y, map2x, map2y)

        # 3.计算视差图
        disp, _ = ss.stereoMatchSGBM(iml_rectified_l, imr_rectified_r, False)

        # 计算像素点的3D坐标（左相机坐标系下）
        points_3d = cv2.reprojectImageTo3D(disp, Q)
        position = np.array([points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2]]) / 1000
        distance = ((points_3d[y, x, 0] ** 2 + points_3d[y, x, 1] ** 2 + points_3d[y, x, 2] ** 2) ** 0.5) / 1000

        return distance, position


if __name__ == '__main__':
    detector = StereoDetector()
    iml = cv2.imread('left_img_40.png')
    imr = cv2.imread('right_img_40.png')
    distance, position = detector.run_detect(iml, imr, 320, 221)
    print('距离：%f, 位置：(%f, %f, %f)' % (distance, position[0], position[1], position[2]))
