import numpy as np
import pandas as pd
import os
import cv2
import joblib
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# GTSRB veri setini yukleme
base_folder = "/Users/haticeaksu/Traffic-Sign-Classification/GTSRB/Training"
data = []
labels = []

# Her sinif klasoru icin islemleri yapma
for class_folder in os.listdir(base_folder):
    class_path = os.path.join(base_folder, class_folder)
    if os.path.isdir(class_path):
        try:
            csv_file = os.path.join(class_path, f"GT-{class_folder}.csv")
            annotations = pd.read_csv(csv_file, sep=';')
            for _, row in annotations.iterrows():
                image_path = os.path.join(class_path, row['Filename'])
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (32, 32))
                    data.append(image)
                    labels.append(row['ClassId'])
        except PermissionError:
            print(f"Permission denied for folder: {class_path}")

# Veriyi numpy array'e donusturme
data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)

# Basit Veri Artirma
augmented_data = []
augmented_labels = []
for i in range(len(data)):
    image = data[i]
    label = labels[i]
    augmented_data.append(image)
    augmented_labels.append(label)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_data.append(rotated_image)
    augmented_labels.append(label)
    flipped_image = cv2.flip(image, 1)
    augmented_data.append(flipped_image)
    augmented_labels.append(label)

augmented_data = np.array(augmented_data, dtype='float32')
augmented_labels = np.array(augmented_labels)

# Veri Setini Train/Test Olarak Ayirma
X_train, X_test, y_train, y_test = train_test_split(augmented_data, augmented_labels, test_size=0.2, random_state=42)

# Torch Dataset Tanimi
class GTSRBDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Veri Dönüşümleri ve Dataloaders
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = GTSRBDataset(X_train, y_train, transform=data_transform)
test_dataset = GTSRBDataset(X_test, y_test, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# RBF Sinir Ağı Modeli Tanımlama
class RBFClassifier(nn.Module):
    def __init__(self):
        super(RBFClassifier, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.rbf_layer = nn.Linear(512, 256)  # RBF katmanı için doğrusal yaklaşım
        self.fc2 = nn.Linear(256, 43)  # 43 sinif

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.rbf_layer(x))
        x = self.fc2(x)
        return x

# Model, Loss Fonksiyonu ve Optimizasyon
model = RBFClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model Eğitimi
num_epochs = 10
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

end_time = time.time()
print(f"Estimated training time: {end_time - start_time:.2f} seconds")

# Modeli Kaydetme
torch.save(model.state_dict(), 'traffic_sign_rbf_classifier.pth')

# Model Değerlendirme
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

# Sonuçları Yazdırma
print("Classification Report:\n", classification_report(y_true, y_pred))
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(base_folder, 'traffic_sign_rbf_confusion_matrix_v2.png'))
plt.close()
