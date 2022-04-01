from glob import glob
import cv2
import os
from utils.cfa import Demosaic

demosaic = Demosaic()

mask_set_path = 'dataset_flower_origin/train'
path_save = 'dataset_flower_cfa/train'

list_classes = glob(mask_set_path + '/*')
try:
    os.mkdir(path_save.split('/')[0])
except:
    pass
try:
    os.mkdir(path_save)
except:
    pass
print('Starting ..')
for clss in list_classes:
    print('Class: ', clss)
    list_image = glob(clss + '/*.*')
    try:
        os.mkdir(os.path.join(path_save, clss.split('/')[-1]))
    except:
        pass
    for i in list_image:
        try:
            image = cv2.imread(i)
            path_clss_img = ''.join(i.split('/')[-2] + '/' + i.split('/')[-1])
            CFA = demosaic.bgr2cfa(image)
            cv2.imwrite(os.path.join(path_save, path_clss_img), CFA.astype('uint8'))
        except:
            continue

