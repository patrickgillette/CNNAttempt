import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SceneDetectionDataset:
    def __init__(self, train_path, test_path, batch_size=256):
        self.transformer = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.train_loader = DataLoader(
            torchvision.datasets.ImageFolder(train_path, transform=self.transformer),
            batch_size=batch_size, shuffle=True
        )
        
        self.test_loader = DataLoader(
            torchvision.datasets.ImageFolder(test_path, transform=self.transformer),
            batch_size=batch_size, shuffle=True
        )
        
        root = pathlib.Path(train_path)
        self.classes = sorted([j.name for j in root.iterdir() if j.is_dir()])

class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        
        self.fc = nn.Linear(in_features=32 * 75 * 75, out_features=num_classes)

    def forward(self, input):
        output = self.pool(self.relu1(self.bn1(self.conv1(input))))
        output = self.relu2(self.conv2(output))
        output = self.relu3(self.bn3(self.conv3(output)))
        output = output.view(-1, 32 * 75 * 75)
        return self.fc(output)

class ModelManager:
    def __init__(self, model_path='best_checkpoint.model', num_classes=6):
        self.model_path = model_path
        self.model = ConvNet(num_classes).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.loss_function = nn.CrossEntropyLoss()
        
        if os.path.exists(self.model_path):
            print("Loading existing model...")
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            print("No pre-trained model found. Training will start from scratch.")

class Trainer:
    def __init__(self, model_manager, dataset, num_epochs=10):
        self.model_manager = model_manager
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.best_accuracy = 0.0

    def train(self):
        train_count = len(glob.glob(train_path + "*/*"))
        test_count = len(glob.glob(test_path + "*/*"))
        
        for epoch in range(self.num_epochs):
            self.model_manager.model.train()
            train_accuracy, train_loss = 0.0, 0.0
            
            for images, labels in self.dataset.train_loader:
                images, labels = images.to(device), labels.to(device)
                
                self.model_manager.optimizer.zero_grad()
                outputs = self.model_manager.model(images)
                loss = self.model_manager.loss_function(outputs, labels)
                loss.backward()
                self.model_manager.optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, prediction = torch.max(outputs.data, 1)
                train_accuracy += int(torch.sum(prediction == labels.data))
            
            train_accuracy /= train_count
            train_loss /= train_count
            
            test_accuracy = self.evaluate(test_count)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            if test_accuracy > self.best_accuracy:
                torch.save(self.model_manager.model.state_dict(), self.model_manager.model_path)
                self.best_accuracy = test_accuracy

    def evaluate(self, test_count):
        self.model_manager.model.eval()
        test_accuracy = 0.0
        
        with torch.no_grad():
            for images, labels in self.dataset.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model_manager.model(images)
                _, prediction = torch.max(outputs.data, 1)
                test_accuracy += int(torch.sum(prediction == labels.data))
        
        return test_accuracy / test_count