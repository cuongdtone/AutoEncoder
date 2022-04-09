from glob import glob
import cv2
import os
from utils.cfa import Demosaic
import yaml

demosaic = Demosaic()

mask_set_path = 'dataset_flower_origin (copy)/train'
path_save = 'dataset_flower_gray/train'
cfa = False

with open('config.yaml', 'r') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

input_size = param['input_size']
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
            if cfa is True:
                CFA = demosaic.bgr2cfa(image)
                cv2.imwrite(os.path.join(path_save, path_clss_img), CFA.astype('uint8'))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(path_save, path_clss_img), cv2.resize(image, (input_size, input_size)))

        except:
            continue

