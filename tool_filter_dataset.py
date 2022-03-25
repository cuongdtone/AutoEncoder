from glob import glob
import cv2
import numpy as np
import os

mask_set_path = 'dataset_flower'
list_classes = glob(mask_set_path + '/*')
def check_mask_is_perfect(image):
    #perfect is no loss region/feature of object
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    perfect = True
    sum_x = np.sum(mask, 0)
    sum_y = np.sum(mask, 1)
    if sum_x[0] + sum_x[-1] + sum_y[0] + sum_y[-1] !=0:
        perfect = False
    return perfect
def crop_obj(image):
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        return image[y1:y2, x1:x2]
    except:
        return image

for clss in list_classes:
    list_image = glob(clss + '/*.png')
    for i in list_image:
        try:
            mask = cv2.imread(i)
            if check_mask_is_perfect(mask):
                mask = crop_obj(mask)
                cv2.imwrite(i, mask)
            else:
                os.remove(i)
            if mask.shape[0]<50 or mask.shape[1] < [50]:
                os.remove(i)

        except:
            continue
        