import torch
from model import AE, image_torch
import cv2
import glob
import matplotlib.pyplot as plt


input_size = 64
model = AE(input_size=input_size)
model.load_state_dict(torch.load('binary.h5', map_location='cpu'))
model.eval()

list_image = glob.glob('dataset_flower_binary/rose/*.png')

cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ori', 480, 480)
cv2.namedWindow('re', cv2.WINDOW_NORMAL)
cv2.resizeWindow('re', 480, 480)

for i in list_image:
    image = cv2.imread(i)
    #image[image<=127] = 127
    x = model.preprocess_image(image)
    y = model(x)
    x_image = image_torch(x, input_size=input_size)
    y_image = image_torch(y, input_size=input_size)

    cv2.imshow('ori', x_image)
    cv2.imshow('re', y_image)
    cv2.waitKey()
