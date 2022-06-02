# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran
# @Time          : 01/06/2022

import torch
from models.model import AE, image_torch
import cv2
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import yaml
from utils.cfa import demosaic
from utils.functions import concat_image
from utils.functions import save_new_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('src/config.yaml', 'r') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)
input_size = param['input_size']
code_size = param['code_size']

model = torch.load('runs/ae.pt', map_location=device)
model.eval()
print("#Parameter: ", sum(p.numel() for p in model.parameters()))
list_image = glob.glob('dataset_flower_origin/train/*/*')
random.shuffle(list_image)
print("#image test: ", len(list_image))

# cv2.namedWindow('Show', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Show', 1280, 720)
images = []
m = 1
n = 1
maps_predict = np.zeros((input_size * 8, input_size * 8, 3), dtype='uint8')
maps_true = np.zeros((input_size * 8, input_size * 8, 3), dtype='uint8')
count = 0
fig = plt.figure()
for i in list_image:
    image = cv2.imread(i)
    # print(image)
    image_cfa = demosaic.bgr2cfa(image)
    x = model.preprocess_image(image_cfa)
    y = model(x)
    x_image = image_torch(x, input_size=input_size)
    y_image = image_torch(y, input_size=input_size)
    x_image = demosaic.cfa2bgr(x_image)
    y_image = demosaic.cfa2bgr(y_image)
    plt.subplot(1, 2, 1)
    plt.imshow(x_image[..., ::-1])
    plt.title('Orinal')
    plt.subplot(1, 2, 2)
    plt.imshow(y_image[..., ::-1])
    plt.title('Recontruct')
    maps_predict[(m-1) * input_size:m * input_size, (n-1) * input_size:n * input_size] = y_image
    maps_true[(m - 1) * input_size:m * input_size, (n - 1) * input_size:n * input_size] = x_image
    n += 1
    if n == 9:
        m += 1
        n = 1
    if m == 9:
        count += 1
        cv2.imwrite(f'imgs/test_org_{count}.jpg', maps_true)
        cv2.imwrite(f'imgs/test_pred_{count}.jpg', maps_predict)
        m = 1
    if count == 3:
        break
    plt.show()
