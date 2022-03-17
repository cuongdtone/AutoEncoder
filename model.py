import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import cv2

def image_torch(x, input_size=32):
    x_re = x.view(input_size, input_size).detach().numpy()
    image_re = x_re*0.5 + 0.5
    image_re = image_re * 255
    image_re = image_re.astype('uint8')
    return image_re

class AE(nn.Module):
    def __init__(self, input_size=32, code_size=9):
        super().__init__()
        # 1024 ==> 9
        self.input_size = input_size
        self.code_size = code_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size * self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, self.code_size)
        )
        # 9 ==> 1024
        self.decoder = nn.Sequential(
            nn.Linear(self.code_size, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_size * self.input_size),
            nn.Sigmoid()
        )
    def preprocess_image(self, gray_image):
        normalize = transforms.Normalize(mean=[0.5],
                                         std=[0.5])
        preprocess = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(int(self.input_size / 0.875)),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            normalize])
        pil_image = Image.fromarray(gray_image)
        x = preprocess(pil_image)
        x = x.view(-1, self.input_size * self.input_size)
        return x
    def forward(self, x):
        #x = self.preprocess_image(gray_image)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
if __name__ == '__main__':

    model = AE()
    gray_image = cv2.imread('2649404904_b7a91991bb_n.png', 0)
    cv2.imshow('ori', gray_image)
    x = model.preprocess_image(gray_image)
    x = model(x)
    print(x.shape)


    image_re = image_torch(x)
    cv2.imshow('re', image_re)

    cv2.waitKey()