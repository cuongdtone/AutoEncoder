from glob import glob
import cv2
import numpy as np
import os

def crop_obj(image, mask):
    #mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
def check_mask_is_perfect(mask):
    #perfect is no loss region/feature of object
    #mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    perfect = True
    sum_x = np.sum(mask, 0)
    sum_y = np.sum(mask, 1)
    if sum_x[0] + sum_x[-1] + sum_y[0] + sum_y[-1] !=0:
        perfect = False
    return perfect
mask_set_path = '/home/cuong/Downloads/dataset_flower_binary/'
origin_set_path = '/home/cuong/Downloads/dataset_flower_origin/'
path_save = 'dataset_flower_crop_with_mask'
try:
    os.mkdir(path_save)
except:
    pass
list_classes = glob(mask_set_path + '/*')
for clss in list_classes:
    list_image = glob(clss + '/*.png')
    try:
        os.mkdir(os.path.join(path_save, clss.split('/')[-1]))
    except:
        pass
    for i in list_image:
        try:
            mask = cv2.imread(i, 0)
            path_clss_img = ''.join(i.split('/')[-2]+'/'+i.split('/')[-1])
            path_ori = os.path.join(origin_set_path, path_clss_img)
            image = cv2.imread(path_ori)
            image = crop_obj(image, mask)
            cv2.imwrite(os.path.join(path_save, path_clss_img), image)

        except:
            continue

