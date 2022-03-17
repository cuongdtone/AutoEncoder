import os
import torch
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from torch import nn, optim
from model import AE

# Define hyperparameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_path = 'dataset_flower_binary/'
input_size = 64
epochs = 60
batch_size = 16
dir_save_model = ''

if __name__ == '__main__':
    model = AE(input_size=input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
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

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, input_size**2).to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(train_loader)
        print("Epoch : {}/{}, loss = {:.4f}".format(epoch + 1, epochs, loss))
    torch.save(model.state_dict(), os.path.join(dir_save_model, 'binary.h5'))