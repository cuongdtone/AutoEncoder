# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran
# @Time          : 01/06/2022

from sklearn.model_selection import train_test_split
from glob import glob
import os
import shutil
import cv2

dataset_path = 'dataset_flower'
out_path = 'dataset_flower_crop_obj'

try:
    os.mkdir(out_path)
    os.mkdir(out_path + '/train')
    os.mkdir(out_path + '/test')
except:
    pass

list_img = glob(dataset_path+'/*/*')
list_class = glob(dataset_path+'/*')

for i in list_class:
    try:
        os.mkdir(out_path+'/train/' + i.split('/')[1])
        os.mkdir(out_path+'/test/' + i.split('/')[1])
    except:
        pass

train, test = train_test_split(list_img)
for i in train:
    ds_path = out_path + '/train/' + '/'.join(i.split('/')[1:])
    cv2.imwrite(ds_path, cv2.imread(i))
for i in test:
    ds_path = out_path + '/test/' + '/'.join(i.split('/')[1:])
    cv2.imwrite(ds_path, cv2.imread(i))
