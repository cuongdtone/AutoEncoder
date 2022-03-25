from glob import glob
import cv2
import os
from utils.cfa import Demosaic

demosaic = Demosaic()

mask_set_path = 'dataset_flower'
path_save = 'dataset_flower_cfa'
list_classes = glob(mask_set_path + '/*')
try:
    os.mkdir(path_save)
except:
    pass
for clss in list_classes:
    list_image = glob(clss + '/*.png')
    try:
        os.mkdir(os.path.join(path_save, clss.split('/')[-1]))
    except:
        pass
    for i in list_image:
        image = cv2.imread(i)
        path_clss_img = ''.join(i.split('/')[-2] + '/' + i.split('/')[-1])
        CFA = demosaic.bgr2cfa(image)
        cv2.imwrite(os.path.join(path_save, path_clss_img), CFA.astype('uint8'))

