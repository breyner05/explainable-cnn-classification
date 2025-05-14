#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:58:45 2025

@author: breyner
"""
###############################PAQUETES######################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from sklearn.model_selection import KFold
import numpy as np
import torchvision.transforms as transforms
############################################################################

# Definir transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4466], std=[0.2454, 0.2397, 0.2501])
])

# Cargar dataset
data_dir = './data/cifar10'
dataset = ImageFolder(data_dir + '/train', transform=transform)

# Definir los hiperparámetros a probar
learning_rates = [0.01, 0.001, 0.0001]
num_layers = [2, 3, 4]  # Número de capas convolucionales
kernel_sizes = [3, 5]   # Tamaños de kernel
num_epochs = 10
batch_size = 64
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Definir la CNN
torch.manual_seed(42)
class CNN(nn.Module):
    def __init__(self, num_classes=10, num_layers=3, kernel_size=3):
        super(CNN, self).__init__()
        
        layers = []
        input_channels = 3
        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, 32 * (i+1), kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())  
            layers.append(nn.MaxPool2d(2, 2))  
            input_channels = 32 * (i+1)
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calcular dinámicamente el tamaño de entrada para la capa totalmente conectada
        self._to_linear = None
        self._compute_linear_size()
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def _compute_linear_size(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 32, 32)  # Simula una imagen de CIFAR-10
            x = self.conv_layers(x)
            self._to_linear = x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # Aplanamos para FC
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función para entrenar un fold
def train_fold(train_loader, val_loader, lr, num_layers, kernel_size):
    model = CNN(num_layers=num_layers, kernel_size=kernel_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluación
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Aplicar K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for num_layer in num_layers:
    for kernel in kernel_sizes:
        for lr in learning_rates:
            fold_accuracies = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                print(f"Fold {fold+1}/5 - Layers: {num_layer}, Kernel: {kernel}, LR: {lr}")
                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
                
                acc = train_fold(train_loader, val_loader, lr, num_layer, kernel)
                fold_accuracies.append(acc)
                
                # Imprimir el accuracy de cada fold
                print(f"  Accuracy en Fold {fold+1}: {acc:.2f}%")
            
            # Calcular el promedio y la desviación estándar
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            results[(num_layer, kernel, lr)] = (mean_acc, std_acc)
            
            # Imprimir el promedio de accuracy
            print(f"Avg Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")

# Seleccionar la mejor configuración
best_params = max(results, key=lambda x: results[x][0])
print(f"\nMejor combinación: Layers = {best_params[0]}, Kernel = {best_params[1]}, LR = {best_params[2]}, Accuracy = {results[best_params][0]:.2f}% ± {results[best_params][1]:.2f}%")
