# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran
# @Time          : 01/06/2022

from torchvision import transforms, datasets

normalize = transforms.Normalize(mean=[0.5],
                                 std=[0.5])
transformer = transforms.Compose([
    transforms.Grayscale(),
    # transforms.Resize(input_size),
    # transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    normalize])