import torch
import os
import model
from torch import nn, optim
from torchvision import models
from load_data import load_train_data


data_dir = 'data/train/'
batch_size = 2
learning_rate = 0.001
epochs = 10
steps = 0
use_pretrain_model = False


train_loader = load_train_data(data_dir, batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# you can choose pytorch's pretrained model or builded by yourself
if use_pretrain_model is True:
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(9216, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, len(os.listdir(data_dir)))
    )
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
elif not use_pretrain_model:
    model = model.AlexNet(num_classes=len(os.listdir(data_dir)))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()
model.to(device)


for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
        model.eval()
        print('epoch:%d' % (epoch+1), 'step:%d' % steps, 'train_loss:%.3f' % running_loss)
        running_loss = 0
        model.train()

if not os.path.exists('save_model'):
    os.makedirs('save_model')
torch.save(model, './save_model/model.pth')
