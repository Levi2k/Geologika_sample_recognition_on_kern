import os, time
# import pathlib
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, increment_path #, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class YOLO(object):
    def __init__(self, weights='yolo_module/data/best.pt', conf_thres=0.4, img_size=640):

        self.conf_thres = 0.6

        self.weights = weights
        self.imgsz = img_size

        self.augment = False
        self.conf_thres = conf_thres
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.save_img = False

        # ---- Initialize
        # set_logging()

        self.device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # ---- Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # ---- Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # ---- Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.classes = None

        # ---- Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def resize_square(self, img, height=416, color=(0, 0, 0)):  # resize a rectangular image to a padded square
        shape = img.shape[:2]  # shape = [height, width]
        ratio = float(height) / max(shape)  # ratio  = old / new
        new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
        dw = height - new_shape[1]  # width padding
        dh = height - new_shape[0]  # height padding
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)  # resized, no border
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw // 2, dh // 2

    def prepare_img(self, img):
        img_boxed, _, _, _ = self.resize_square(img.copy(), height=self.imgsz, color=(127.5, 127.5, 127.5))
        img_processed = img_boxed.copy()[:, :, ::-1].transpose(2, 0, 1)
        return img_processed

    def detect(self, img):
        result_chopped_imgs = []

        im0s = img.copy()
        img_processed = self.prepare_img(img)
        img = img_processed.copy()

        t0 = time.time()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # ---- Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # ---- Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()

        # ---- Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0s)
        path=None
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            # p = pathlib.Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            copy_im0 = np.copy(im0)
            # if len(det):
                # ---- Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # # ---- Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # # ---- Write results
                # for *xyxy, conf, cls in reversed(det):
                #     if self.save_img:  # Add bbox to image
                #         label = f'{self.names[int(cls)]} {conf:.2f}'
                #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # ---- Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')             

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x,y,w,h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]),   int(xyxy[3] - xyxy[1])
                    # img_ = im0.astype(np.uint8)
                    crop_img = copy_im0[y:y+h,x:x+w] 
                    # cv2.imwrite(save_path.replace('.jpg', '_%d.jpg' %  i), crop_img)
                    entry = {
                        'img':crop_img, 
                        'bbox':[x,y,w,h],
                        'conf':float(conf.cpu().detach().numpy())
                        }
                    result_chopped_imgs.append(entry)

        return result_chopped_imgs      
        # return np.array(boxes), np.array(scores), np.array(classes)


if __name__ == '__main__':
    pass