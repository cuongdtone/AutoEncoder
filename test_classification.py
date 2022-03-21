import torch
from torch import nn
from model import AE_NET, image_torch
import cv2
import glob
import matplotlib.pyplot as plt


input_size = 100
model = AE_NET(input_size=input_size, code_size=30)
model.load_state_dict(torch.load('classifier_binary.h5', map_location='cpu'))
model.eval()

list_image = glob.glob('dataset_flower_binary/sunflower/*.png')

cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ori', 480, 480)
cv2.namedWindow('re', cv2.WINDOW_NORMAL)
cv2.resizeWindow('re', 480, 480)

for i in list_image:
    image = cv2.imread(i)
    #image[image<=127] = 127
    x = model.feature_extractor.preprocess_image(image)
    out = model(x)
    _, index = torch.max(out, 1)
    percentage = (nn.functional.softmax(out, dim=1)[0] * 100).tolist()

    print(index, max(percentage))

    cv2.imshow('ori', image)
    cv2.waitKey()