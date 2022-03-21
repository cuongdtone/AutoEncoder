from glob import glob
import cv2
import numpy as np
import os
import random
def crop_obj(mask):
    sum_x = np.sum(mask, 0)
    sum_y = np.sum(mask, 1)
    for idx in range(len(sum_x)-1):
        if sum_x[idx]==0 and sum_x[idx+1]!=0:
            x1 = idx+1
        if sum_x[idx]!=0 and sum_x[idx+1]==0:
            x2 = idx+1
    for idx in range(len(sum_y)-1):
        if sum_y[idx]==0 and sum_y[idx+1]!=0:
            y1 = idx+1
        if sum_y[idx]!=0 and sum_y[idx+1]==0:
            y2 = idx+1
    try:
        return mask[y1:y2, x1:x2]
    except:
        return mask
mask_set_path = 'dataset_flower_gray'
list_classes = glob(mask_set_path + '/*')
for clss in list_classes:
    list_image = glob(clss + '/*.png')
    for i in list_image:
        image = cv2.imread(i)


