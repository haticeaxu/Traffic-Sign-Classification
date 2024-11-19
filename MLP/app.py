import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os
import cv2
import joblib
import time

# GTSRB veri setini yükleme
# Veri klasörünün yolu
base_folder = "/Users/haticeaksu/Traffic-Sign-Classification/GTSRB/Training"
data = []
labels = []

# Her sınıf klasörü için işlemleri yapma
for class_folder in os.listdir(base_folder):
    class_path = os.path.join(base_folder, class_folder)
    if os.path.isdir(class_path):
        try:
            # CSV dosyasını yükleme
            csv_file = os.path.join(class_path, f"GT-{class_folder}.csv")
            annotations = pd.read_csv(csv_file, sep=';')
            
            # Görüntüleri yükleyip işleme
            for _, row in annotations.iterrows():
                image_path = os.path.join(class_path, row['Filename'])
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (32, 32))  # Tüm görüntüleri aynı boyuta getiriyoruz (32x32)
                    data.append(image)
                    labels.append(row['ClassId'])
        except PermissionError:
            print(f"Permission denied for folder: {class_path}")

# Veriyi numpy array'e dönüştürme
data = np.array(data)
labels = np.array(labels)

# Veriyi normalize etme (0-1 aralığına)
data = data.astype('float32') / 255.0

# Basit Veri Artırma (Data Augmentation)
augmented_data = []
augmented_labels = []
for i in range(len(data)):
    image = data[i]
    label = labels[i]
    augmented_data.append(image)
    augmented_labels.append(label)
    
    # Görüntüyü döndürme
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_data.append(rotated_image)
    augmented_labels.append(label)
    
    # Görüntüyü yatay çevirme
    flipped_image = cv2.flip(image, 1)
    augmented_data.append(flipped_image)
    augmented_labels.append(label)

# Veriyi numpy array'e dönüştürme
augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)

# Veriyi Pandas DataFrame olarak düzenleme
# Özellikleri düzleştiriyoruz çünkü MLP modeline uygun formatta olması gerekiyor
num_samples = augmented_data.shape[0]
num_features = augmented_data.shape[1] * augmented_data.shape[2] * augmented_data.shape[3]
data_flat = augmented_data.reshape((num_samples, num_features))

veriler = pd.DataFrame(data_flat)
veriler['Label'] = augmented_labels

# Eksik Değerlerin İşlenmesi (Eksik değerleri doldurma)
veriler.fillna(veriler.mean(), inplace=True)

# Veri Setini Eğitim ve Test Olarak Ayırma
X = veriler.drop('Label', axis=1)
y = veriler['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellik Ölçekleme (Standartleştirma)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLP Modeli (Daha karmaşık bir yapı ile)
mlp = MLPClassifier(max_iter=150, verbose=True)
parameter_space = {
    'hidden_layer_sizes': [(100, 100), (150, 100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
}

# Eğitim süresini tahmin etmek için zaman ölçümü
start_time = time.time()
clf = GridSearchCV(mlp, parameter_space, n_jobs=3, cv=3, verbose=3)
clf.fit(X_train_scaled, y_train)
end_time = time.time()

# Eğitim süresini hesaplama
estimated_eta = end_time - start_time
print(f"Estimated training time: {estimated_eta:.2f} seconds")

# En iyi modeli kaydetme
best_mlp_model = clf.best_estimator_
joblib.dump(best_mlp_model, 'best_mlp_model.pkl')

# MLP Sonuçları
y_pred_mlp = best_mlp_model.predict(X_test_scaled)
print("MLP Classification Report:\n", classification_report(y_test, y_pred_mlp))
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_mlp, annot=True, cmap='Blues', fmt='g')
plt.title(f"MLP Confusion Matrix\nBest Parameters: {clf.best_params_}")
plt.savefig(os.path.join(base_folder, 'mlp_confusion_matrix_v2.png'))
plt.close()
