from glob import glob
import cv2
import numpy as np
import os

mask_set_path = '/home/cuong/Downloads/dataset_flower_binary/'
origin_set_path = '/home/cuong/Downloads/dataset_flower_origin/'
path_save = 'dataset_flower'
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
            image[mask==0] = [0, 0, 0]
            cv2.imwrite(os.path.join(path_save, path_clss_img), image)

        except:
            continue

