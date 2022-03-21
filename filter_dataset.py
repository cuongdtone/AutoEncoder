from glob import glob
import cv2
import numpy as np
import os

mask_set_path = 'dataset_flower_gray'
list_classes = glob(mask_set_path + '/*')
def check_mask_is_perfect(mask):
    #perfect is no loss region/feature of object
    perfect = True
    sum_x = np.sum(mask, 0)
    sum_y = np.sum(mask, 1)
    if sum_x[0] + sum_x[-1] + sum_y[0] + sum_y[-1] !=0:
        perfect = False
    return perfect
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

for clss in list_classes:
    list_image = glob(clss + '/*.png')
    for i in list_image:
        try:
            mask = cv2.imread(i, 0)
            if check_mask_is_perfect(mask):
                mask = crop_obj(mask)
                cv2.imwrite(i, mask)
            else:
                os.remove(i)
            if mask.shape[0]<50 or mask.shape[1] < [50]:
                os.remove(i)

        except:
            continue
        