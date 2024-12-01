import random
import joblib
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

# Sınıf isimlerini dinamik olarak yükleme
base_folder = "/Users/haticeaksu/Traffic-Sign-Classification/GTSRB/Training"
class_names = {
    0: 'Hız sınırı (20 km/h)',
    1: 'Hız sınırı (30 km/h)',
    2: 'Hız sınırı (50 km/h)',
    3: 'Hız sınırı (60 km/h)',
    4: 'Hız sınırı (70 km/h)',
    5: 'Hız sınırı (80 km/h)',
    6: 'Hız sınırı sonu (80 km/h)',
    7: 'Hız sınırı (100 km/h)',
    8: 'Hız sınırı (120 km/h)',
    9: 'Geçiş yok',
    10: 'Kamyonlar için geçiş yok',
    11: 'Öncelikli yol',
    12: 'Yol ver',
    13: 'Dur',
    14: 'Geçiş yasak',
    15: 'Kamyonlar için geçiş yasak',
    16: 'Girilmez',
    17: 'Dikkat',
    18: 'Tehlike',
    19: 'Sağa viraj',
    20: 'Sola viraj',
    21: 'S-viraj',
    22: 'Kaygan yol',
    23: 'Yol daralması',
    24: 'Yol çalışması',
    25: 'Trafik ışıkları',
    26: 'Yaya geçidi',
    27: 'Çocuk geçidi',
    28: 'Bisiklet geçidi',
    29: 'Dikkat hayvan',
    30: 'Tehlikeli kenar',
    31: 'Dönel kavşak',
    32: 'Geçiş yasak sonu',
    33: 'Hız sınırı sonu (120 km/h)'
}

# Kaydedilen en iyi modeli yükleme
model_path = 'best_mlp_model_v2.pkl'
best_mlp_model = joblib.load(model_path)

# Tüm sınıflardan rastgele bir görüntü seçme ve tahmin yapma
for class_id, class_name in class_names.items():
    class_folder_name = f'{class_id:05d}'  # Klasör isimleri sıfırla başlatılmış olabilir (ör: 00000, 00001)
    class_path = os.path.join(base_folder, class_folder_name)

    # Klasörün var olup olmadığını kontrol etme
    if not os.path.isdir(class_path):
        continue

    csv_file = os.path.join(class_path, f"GT-{class_folder_name}.csv")
    if not os.path.exists(csv_file):
        continue

    annotations = pd.read_csv(csv_file, sep=';')

    random_image_row = annotations.sample(n=1).iloc[0]
    random_image_path = os.path.join(class_path, random_image_row['Filename'])
    random_image = cv2.imread(random_image_path)

    if random_image is not None:
        # Görüntü boyutunu yeniden ayarlama (32x32)
        random_image_resized = cv2.resize(random_image, (32, 32))
        random_image_normalized = random_image_resized.astype('float32') / 255.0

        # Görüntüyü model için uygun formata getirme (düzleştirme)
        random_image_flat = random_image_normalized.reshape(1, -1)

        # Tahmin yapma
        predicted_label = best_mlp_model.predict(random_image_flat)[0]
        predicted_class_name = class_names.get(predicted_label, f'Bilinmeyen Sınıf ({predicted_label})')
        print(f"Seçilen rastgele görüntünün tahmin edilen sınıfı: {predicted_label} - {predicted_class_name}")

        # Görüntüyü açma
        plt.imshow(cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Tahmin Edilen Sınıf: {predicted_class_name}")
        plt.axis('off')
        plt.show()
    else:
        print(f"Görüntü yüklenemedi! Sınıf: {class_name}")