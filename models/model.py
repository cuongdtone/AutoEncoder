# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran
# @Time          : 01/06/2022


import torch
from torch import nn
from PIL import Image
import sys
from pathlib import Path
from utils.transform import transformer


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def image_torch(x, input_size=32):
    x_re = x.view(input_size, input_size).detach().numpy()
    image_re = x_re*0.5 + 0.5
    image_re = image_re * 255
    image_re = image_re.astype('uint8')
    return image_re


class AE(nn.Module):
    def __init__(self, input_size=32, code_size=15):
        super().__init__()
        # 1024 ==> 9
        self.input_size = input_size
        self.code_size = code_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size **2, 2056),
            nn.Hardswish(inplace=True),
            ##
            nn.Linear(2056, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(inplace=True),
            ##
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(inplace=True),
            ##
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Hardswish(inplace=True),
            ##
            nn.Linear(64, code_size))
            ##
        self.decoder = nn.Sequential(
            nn.Linear(code_size, 64),
            nn.BatchNorm1d(64),
            nn.Hardswish(inplace=True),
            ##
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(inplace=True),
            ##
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(inplace=True),
            ##
            nn.Linear(1024, 2056),
            nn.Hardswish(inplace=True),
            ##
            nn.Linear(2056, input_size ** 2),
            nn.Tanh())

    def preprocess_image(self, gray_image):
            preprocess = transformer
            pil_image = Image.fromarray(gray_image)
            x = preprocess(pil_image)
            x = x.view(-1, self.input_size * self.input_size)
            return x

    def forward(self, x):
        #x = self.preprocess_image(gray_image)
        encoded = self.encoder(x)
        #print(encoded)
        decoded = self.decoder(encoded)
        return decoded

    def get_coding(self, x):
        return self.encoder(x)

    def decode(self, code):
        return self.decoder(code)

class Net(nn.Module):

    def __init__(self, input_size=32, num_classes=4, device='cpu'):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        out = self.classifier(x)
        return out


class AE_NET(nn.Module):

    def __init__(self, num_classes=5, input_size=100, code_size=30, feature_etractor='model.h5', device='cpu'):
        super(AE_NET, self).__init__()
        self.feature_extractor = AE(input_size=input_size, code_size=code_size)
        if feature_etractor != '':
            self.feature_extractor.load_state_dict(torch.load(feature_etractor, map_location=device))
            self.feature_extractor.eval()
        self.classifier = nn.Sequential(
            nn.Linear(code_size, 15),
            nn.ReLU(),
            nn.Linear(15, num_classes),
        )

    def forward(self, x):
        feature = self.feature_extractor.get_coding(x)
        out = self.classifier(feature)
        return out


if __name__ == '__main__':
    net = Net()
    x = torch.rand((124, 32))
    out = net(x)
    print(x.shape)
    print(out.shape)