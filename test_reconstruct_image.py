import torch
from models.model import AE, image_torch
import cv2
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import yaml
from utils.cfa import Demosaic
from utils.draw import concat_image
from utils.output import save_new_image


with open('config.yaml', 'r') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)
input_size = param['input_size']
code_size = param['code_size']

model = AE(input_size=input_size, code_size=param['code_size'])
model.load_state_dict(torch.load('runs/ae.pt', map_location='cpu'))
model.eval()
print("#Parameter: ", sum(p.numel() for p in model.parameters()))
list_image = glob.glob('dataset_flower_cfa/test/*/*')
random.shuffle(list_image)
print("#image test: ", len(list_image))

demosaic = Demosaic()
cv2.namedWindow('Show', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Show', 1280, 720)
images = []
for i in list_image:
    image = cv2.imread(i)

    x = model.preprocess_image(image)
    y = model(x)
    x_image = image_torch(x, input_size=input_size)
    y_image = image_torch(y, input_size=input_size)

    x_image = demosaic.cfa2bgr(x_image)
    y_image = demosaic.cfa2bgr(y_image)
    one_case = cv2.vconcat([x_image, y_image])
    images.append(one_case)
    if len(images) == 16:
        show = concat_image(images, grid_shape=(2, 8), image_size=(input_size, input_size*2))
        images = []
        save_new_image('imgs', show)
        cv2.imshow('Show', show)
        key = cv2.waitKey(5)
        while key != ord(' '):
            key = cv2.waitKey(5)

show = concat_image(images, grid_shape=(2, 8))
images = []
cv2.imshow('Show', show)
cv2.waitKey()
