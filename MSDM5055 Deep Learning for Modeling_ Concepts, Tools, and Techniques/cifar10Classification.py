import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import sys

batch_size = 128

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, groups=3),
            nn.Conv2d(3, 36, 1),  # 增加到36
            nn.ReLU(),
            nn.BatchNorm2d(36),
            # nn.Dropout2d(0.1), 
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(36, 36, 3, padding=1, groups=36),
            nn.Conv2d(36, 96, 1),  # 增加到96
            nn.ReLU(),
            nn.BatchNorm2d(96),
            # nn.Dropout2d(0.1), 
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, groups=96),
            nn.Conv2d(96, 128, 1),  # 增加到128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.1), 
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
NeuralNetwork.__module__ = "cifar10Classification"

import torch.serialization

torch.serialization.add_safe_globals([
    NeuralNetwork,
    nn.Sequential,
    nn.Linear,
    nn.ReLU,
    nn.Dropout,
    nn.Conv2d,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.BatchNorm2d
])

def train(net):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = 40

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1} loss: {running_loss / len(train_loader):.3f}")
        evaluate(net)
        scheduler.step()

def evaluate(net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    sys.modules["cifar10Classification"] = sys.modules["__main__"]
    net = NeuralNetwork()
    train(net)
    torch.save(net, 'cifarNet.pth')