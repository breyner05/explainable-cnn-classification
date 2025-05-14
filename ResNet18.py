#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 22:35:28 2025

@author: breyner
"""

###############################PAQUETES######################################
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

############################################################################


batch_size=64
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),  # 30% de probabilidad de voltear horizontalmente
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Pequeñas traslaciones (máx. 5% de la imagen)
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Variación leve en brillo y contraste
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4466], std=[0.2454, 0.2397, 0.2501]),
    transforms.Resize((224, 224))
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4466], std=[0.2454, 0.2397, 0.2501]),
    transforms.Resize((224, 224))
])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Cargar dataset
data_dir = '/Users/breyner/Desktop/data/cifar10'
train_dataset = ImageFolder(data_dir + '/train', transform=transform)
test_dataset = ImageFolder(data_dir + '/test', transform=transform2)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modelo ResNet18 con Dropout en la última capa
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout del 30% en la capa totalmente conectada
    nn.Linear(model.fc.in_features, 10)
)

model = torch.nn.DataParallel(model)
model = model.to(device)

# Hiperparámetros
num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Entrenamiento
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()  # Ajustar la tasa de aprendizaje al final de la época

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {epoch_loss:.4f}, Precisión: {epoch_accuracy:.2f}%")

# Evaluación
model.eval()
all_preds, all_labels = [], []
total_loss, correct, total = 0.0, 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Cálculo de métricas
test_accuracy = 100 * correct / total
average_loss = total_loss / len(test_loader)

print(f"Accuracy en prueba: {test_accuracy:.2f}%")
print(f"Cross-Entropy Loss en prueba: {average_loss:.4f}")

# Matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
conf_matrix = confusion_matrix(all_labels, all_preds)

print("Reporte de clasificación:")
print(classification_report(all_labels, all_preds, digits=4))


torch.save(model.state_dict(), "resnet18_cifar10.pth")

###################################sección para cargar el modelo que ya fue entrenado
# Crear el modelo de nuevo
def convert_relu_inplace(module):
    for child_name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU) and child.inplace:
            setattr(module, child_name, torch.nn.ReLU(inplace=False))
        else:
            convert_relu_inplace(child)
            
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 10)
)

model = torch.nn.DataParallel(model)
model = model.to(device)

# Cargar los pesos guardados
model.load_state_dict(torch.load("/Users/breyner/Desktop/resnet18_cifar10_64.pth", map_location=device))
model.eval()  # Poner en modo evaluación
print("Modelo cargado correctamente")
convert_relu_inplace(model.module)
#####################################



##########LIME
from PIL import Image
import torch.nn.functional as F


def get_input_transform():
    normalize =  transforms.Normalize(mean=[0.4914, 0.4822, 0.4466], std=[0.2454, 0.2397, 0.2501])    
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((224, 224))
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4466], std=[0.2454, 0.2397, 0.2501])   
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf 
pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()
def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

from lime import lime_image
explainer = lime_image.LimeImageExplainer()
img1 = Image.open('/Users/breyner/Desktop/data/cifar10/test/airplane/0099.png').resize(224,224)
explanation = explainer.explain_instance(np.array(pill_transf(img1)), 
                                         batch_predict, # classification function
                                         top_labels=5, 
                                         hide_color=0, 
                                         num_samples=2500)

image_indices = [
    '0351', '0780', '0874', '0602', '0317', '0617', '0592', '0235', '0601', '0468',
    '0099', '0133', '0668', '0369', '0618', '0516', '0888', '0601', '0402', '0602'
]

image_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]
explanations = []
start = time.perf_counter()
for cls, idx in zip(image_classes, image_indices):
    img_path = f'/Users/breyner/Desktop/data/cifar10/test/{cls}/{idx}.png'
    img1 = Image.open(img_path).resize((224, 224))
    
    # Explicación
    explanation = explainer.explain_instance(
        np.array(pill_transf(img1)), 
        batch_predict, 
        top_labels=5, 
        hide_color=0, 
        num_samples=1000
    )
    
    # Almacenamos la explicación
    explanations.append(explanation)
    
    # Obtener la imagen y la máscara para los contornos
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    
    # Dibujar los contornos sobre la imagen
    img_boundry2 = mark_boundaries(temp/255.0, mask)
    
    # Mostrar la imagen con contornos
    plt.imshow(img_boundry2)
    plt.show()

end = time.perf_counter()

print(f"El bloque tomó {end - start:.6f} segundos")

from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)
plt.show()


temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=50, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)
plt.show()


#####shap
class_names = {
    0: "avión",
    1: "automóvil",
    2: "pájaro",
    3: "gato",
    4: "ciervo",
    5: "perro",
    6: "rana",
    7: "caballo",
    8: "barco",
    9: "camión"
}
import random
import shap
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

# --- Preparar background para SHAP ---
random_indices = random.sample(range(len(test_dataset)), 500)
background = torch.stack([test_dataset[i][0].to(device).clone() for i in random_indices])




####Variar x para obtener imagen a cual explicar
x = [100:112]
img_tensors = torch.stack([test_dataset[i][0] for i in range(9110,9118) ]).to(device)

start = time.perf_counter()
explainer_shap = shap.GradientExplainer((model.module, model.module.layer3), background)
shap_values, indexes = explainer_shap.shap_values(img_tensors, ranked_outputs=1, nsamples=1000)

index_names = np.vectorize(lambda x: class_names[x])(indexes.cpu().numpy())


# --- Desnormalizar imagen para visualizar ---
def denormalize(img_tensor, mean, std):
    # Desnormaliza
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    # Clipa entre 0 y 1
    return img_tensor.clamp(0, 1)

img_tensor_denorm = denormalize(img_tensors.clone(), [0.4914, 0.4822, 0.4466], [0.2454, 0.2397, 0.2501])
img_numpy = img_tensor_denorm.permute(0, 2, 3, 1).cpu().numpy() * 255.0
img_numpy = np.clip(img_numpy, 0, 255).astype(np.uint8)
# --- Visualizar TODAS las clases ---
shap_numpy = list(np.transpose(shap_values, (4, 0, 2, 3, 1)))

shap.image_plot(shap_numpy, img_numpy,index_names)

end = time.perf_counter()

print(f"El bloque tomó {end - start:.6f} segundos")





