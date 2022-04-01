import torch
from PIL import Image
from utils.transform import transformer
class custom_dataset(torch.utils.data.Dataset):
    # 가져와서 처리
    def __init__(self, input_list, output_list, transform=transformer):
        self.input_list = input_list
        self.output_list = output_list
        self.transform = transform

    # dataset length
    def __len__(self):
        self.filelength = len(self.output_list)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_path = self.input_list[idx]
        img_out_path = self.output_list[idx]
        img_input_path = img_out_path.split('/')[-2:-1]

        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0
        return img_transformed, label