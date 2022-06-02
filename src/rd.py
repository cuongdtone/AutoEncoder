import glob

import cv2
import torch
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from torch import nn, optim
from models.model import AE, image_torch
import yaml
import colour
from utils.cfa import Demosaic
from utils.transform import transformer as preprocess
from utils.dataset import custom_dataset

with open('config.yaml', 'r') as f:
    param = yaml.load(f, yaml.FullLoader)

demosaic = Demosaic()
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_path = 'dataset_flower_cfa_2'
label_path = '../dataset_flower_cfa'

input_size = param['input_size']

file_list = glob.glob(input_path + '/*/*')
dataset = custom_dataset(file_list)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ori', 1280, 720)
for batch_features, _ in train_loader:
    batch_features = batch_features.view(-1, input_size**2).to(device)[0]
    x_image = image_torch(batch_features, input_size)
    recontruct = demosaic.cfa2bgr(x_image)
    cv2.imshow('ori', x_image)
    cv2.imshow('recontruc', recontruct)
    cv2.waitKey()