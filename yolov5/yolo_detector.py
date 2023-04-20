import cv2
import numpy as np
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device


class Yolo():
    def __init__(self):
        self.img_size = 640
        self.stride = 32

        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = attempt_load('yolov5/runs/train/exp4/weights/best.pt', map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size

        if self.half:
            self.model.half()  # to FP16

        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def run_detection(self, img0):
        img0 = np.ascontiguousarray(img0)
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.5, 0.4)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    return [list(map(int, [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()])), conf, cls]

        return [0, 0, 0]


if __name__ == '__main__':
    detector = Yolo()
    img_right = cv2.imread("right_img_40.png")
    xyxy, conf, cls = detector.run_detection(img_right)
    cv2.rectangle(img_right, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), [0, 0, 255])
    x = (xyxy[0] + xyxy[2]) / 2
    y = (xyxy[1] + xyxy[3]) / 2
    print(x)
    print(y)
    cv2.imshow("img_right", img_right)
    cv2.waitKey()
    img_left = cv2.imread("img.png")
    xyxy, conf, cls = detector.run_detection(img_left)
    cv2.rectangle(img_left, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), [0, 0, 255])
    x = (xyxy[0] + xyxy[2])/2
    y = (xyxy[1] + xyxy[3])/2
    print(x)
    print(y)
    # cv2.circle(img_left, (int(x), int(y)), radius=2, color=[0, 0, 255])
    cv2.imshow("img_left", img_left)
    cv2.waitKey()

