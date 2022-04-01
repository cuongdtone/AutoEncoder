import os
import torch
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from torch import nn, optim
from models.model import AE, image_torch
from utils.transform import transformer
import yaml
import time

# Define hyperparameter
input_path = 'dataset_flower_cfa/train'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('config.yaml', 'r') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

input_size = param['input_size']
code_size = param['code_size']
epochs = 300
batch_size = 128
dir_save_model = 'runs'

if __name__ == '__main__':
    model = AE(input_size=input_size, code_size=code_size).to(device)
    print("#Parameter: ", sum(p.numel() for p in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    preprocess = transformer

    dataset = datasets.ImageFolder(input_path, preprocess)
    print('Total images: ', len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Start traning ...")
    for epoch in range(epochs):
        last_time = time.time()
        loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, input_size**2).to(device) #flatten
            optimizer.zero_grad() # grad = 0
            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(train_loader)
        total_memory, used_memory_after, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        print("Epoch : {}/{}, loss = {:.4f}, time = {:.1f}, Memory = {:d}/{:d} MB".format(epoch + 1, epochs, loss, time.time()-last_time, used_memory_after, total_memory))
    torch.save(model.state_dict(), os.path.join(dir_save_model, 'ae.pt'))