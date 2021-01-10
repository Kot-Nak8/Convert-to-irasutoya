import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import cv2
import numpy as np


def convert_i(all_box, all_label, image):
    for i in range(len(all_label)):
        if i == 20:
            break
        x1 = all_box[i][0]
        y1 = all_box[i][1]
        x2 = all_box[i][2]
        y2 = all_box[i][3]
        la = all_label[i] 
        w = x2 - x1
        h = y2 - y1
        irasuto = cv2.imread("irasuto/" + la + ".png", cv2.IMREAD_UNCHANGED)
        irasuto = cv2.resize(irasuto, (w,h))
        image[y1:y2, x1:x2] = image[y1:y2, x1:x2] * (1 - irasuto[:, :, 3:] / 255) + irasuto[:, :, :3] * (irasuto[:, :, 3:] / 255)

    return image




if __name__ == '__main__':
    #変換したい画像
    img = "test.jpg"
    #背景に使用する画像
    bg = cv2.imread("bg/bg1.png")
    yolo = YOLO()
    image = Image.open(img)
    r_image, all_box, all_label  = yolo.detect_image(image)
    yolo.close_session()
    img = np.asarray(image)
    bgh = img.shape[0]
    bgw = img.shape[1]
    bg = cv2.resize(bg, (bgw,bgh))
    result = convert_i(all_box, all_label, bg)
    cv2.imwrite("result.png", result)

