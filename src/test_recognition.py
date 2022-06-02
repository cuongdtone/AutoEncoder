# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran
# @Time          : 01/06/2022

import torch
from torch import nn
from models.model import AE_NET, image_torch, AE
from utils.cfa import Demosaic
import cv2
import glob
import matplotlib.pyplot as plt
import yaml
import random
from utils.plot import plot_cm
from sklearn import metrics

with open('config.yaml', 'r') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

input_size = param['input_size']
code_size = param['code_size']

model = AE_NET(input_size=input_size, code_size=code_size, feature_etractor='')
model.load_state_dict(torch.load('runs/classifier.h5', map_location='cpu'))
model.eval()

ae = AE(input_size, code_size)
ae.load_state_dict(torch.load('runs/ae.pt', map_location='cpu'))
ae.eval()

demosaic = Demosaic()

print("#Parameter: ", sum(p.numel() for p in model.parameters()))

list_image = glob.glob('dataset_flower_cfa/test/*/*.png')
random.shuffle(list_image)

with open('../runs/label.txt', 'r') as f:
    class_name = f.readlines()
    class_name = {int(i.split(':')[1].strip('\n')) : i.split(':')[0].strip() for i in class_name}
    class_idx = dict((v, k) for k, v in class_name.items())

# print(class_name)
# print(class_idx)

cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ori', 480, 480)

cv2.namedWindow('recontruct', cv2.WINDOW_NORMAL)
cv2.resizeWindow('recontruct', 480, 480)


for i in list_image:
    image = cv2.imread(i, 0)
    #image[image<=127] = 127
    x = model.feature_extractor.preprocess_image(image)
    out = model(x)
    _, index = torch.max(out, 1)
    percentage = (nn.functional.softmax(out, dim=1)[0] * 100).tolist()
    truth_clss = i.split('/')[-2]
    print('-'*20)
    print('Predict: ', class_name[index.tolist()[0]], ': %.2f %%'%(max(percentage)))
    print('Truth: ', truth_clss)

    y = ae(x)
    recontruct_img = image_torch(y, input_size)

    recontruct_img = demosaic.cfa2bgr(recontruct_img)
    image = demosaic.cfa2bgr(image)
    cv2.imshow('ori', image)
    cv2.imshow('recontruct', recontruct_img)
    cv2.waitKey()
