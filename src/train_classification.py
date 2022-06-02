# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran
# @Time          : 01/06/2022

import os
import torch
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from torch import nn, optim
from models.model import AE, Net
import yaml
from utils.transform import transformer as preprocess
from sklearn.model_selection import train_test_split

# Define hyperparameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_path = 'dataset_flower_gray/train'

with open('config.yaml', 'r') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)
input_size = param['input_size']
code_size = param['code_size']
epochs = 100
batch_size = 32
dir_save_model = '../runs'

if __name__ == '__main__':
    feature_extractor = AE(input_size=input_size, code_size=param['code_size'])
    feature_extractor.load_state_dict(torch.load('runs/ae_gray.pt', map_location='cpu'))
    feature_extractor.eval()
    net = Net(input_size=code_size, num_classes=5)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = datasets.ImageFolder(input_path, preprocess)
    print('Total image: ', len(dataset))
    file = open('../runs/label.txt', "w")
    class_id = dataset.class_to_idx
    for key, value in class_id.items():
        file.write(key + " : " + str(value) + '\n')
    file.close()

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        loss = 0
        running_corrects = 0
        for batch_features, labels in train_loader:
            batch_features = batch_features.view(-1, input_size**2).to(device)
            feature = feature_extractor.get_coding(batch_features)

            outputs = net(feature)
            train_loss = criterion(outputs, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        loss = loss / len(train_loader)
        epoch_acc = running_corrects.double() / len(dataset)
        print("Epoch : {}/{}, loss = {:.4f}, acc = {:.2f}".format(epoch + 1, epochs, loss, epoch_acc))

        torch.save(net.state_dict(), os.path.join(dir_save_model, 'classifier_gray.pt'))

