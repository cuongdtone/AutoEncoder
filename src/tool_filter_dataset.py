from glob import glob
import cv2
import numpy as np
import os
import random
from scipy.ndimage import rotate

def change_brightness(img, value=30):
   '''Truyen vao img, value <0: giam do sang, value >0 : tang do sang '''
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   h, s, v = cv2.split(hsv)
   v = cv2.add(v,value)
   v[v > 255] = 255
   v[v < 0] = 0
   final_hsv = cv2.merge((h, s, v))
   img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
   return img

mask_set_path = 'dataset_flower_origin (copy)'
list_images = glob(mask_set_path + '/train/*/*')

angles = [90, 180, 270]

for i in list_images:
    img = cv2.imread(i)
     # task 1: equa histogram
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # task 2: random rotation
    angle = angles[random.randint(0, 2)]
    angled_img = rotate(img, angle)
    if random.randint(0, 1) == 1:
        angled_img = cv2.blur(angled_img, (3, 3))
    angled_img = change_brightness(angled_img, random.randint(-30, 30))

    path = '.'.join(i.split('.')[:-1]) + '_1.png'
    cv2.imwrite(i, img_output)
    cv2.imwrite(path, angled_img)
    # cv2.imshow('a_img', angled_img)
    # cv2.imshow('image', img_output)
    # cv2.waitKey()

        