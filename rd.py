import cv2
import torch
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from torch import nn, optim
from model import AE, image_torch


#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_path = '/home/cuong/Downloads/dataset_flower_origin/'
input_size = 32

normalize = transforms.Normalize(mean=[0.5],
                                 std=[0.5])
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(int(input_size / 0.875)),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    normalize])

dataset = datasets.ImageFolder(input_path,
                     preprocess)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ori', 1280, 720)
for batch_features, _ in train_loader:
    batch_features = batch_features.view(-1, input_size**2).to(device)[0]
    x_image = image_torch(batch_features, input_size)
    cv2.imshow('ori', x_image)
    cv2.waitKey()