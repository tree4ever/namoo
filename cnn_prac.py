# CNN: Micro-Doppler 스펙트로그램 기반 행동 분류
#   - 레이더로 측정한 사람, 드론, 차량 등은 고유한 Micro-Doppler 패턴을 가집니다. 이를 2D 이미지(스펙트로그램)로 변환하여 CNN으로 분류
#   - 실습 주제: 사람(걷기), 차량(전진), 드론(호버링)의 Micro-Doppler 스펙트로그램을 보고 어떤 타겟인지 분류하는 CNN 모델
#   - 데이터 생성 원리: 걷는 사람은 주기적인 사인파, 차량은 일정한 선, 드론은 복잡한 주파수 형태로 스펙트로그램을 생성

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 가상 Micro-Doppler 스펙트로그램 데이터 생성
def generate_spectrograms(n_samples=100, img_size=64):
    images, labels = [], []
    for i in range(n_samples):
        # Class 0: 사람 (걷기) - Sinusoidal pattern
        if i % 3 == 0:
            t = np.linspace(0, np.pi * 2, img_size)
            freq = np.sin(t * np.random.uniform(2, 4)) * (img_size / 4) + (img_size / 2)
            img = np.zeros((img_size, img_size))
            for j, f in enumerate(freq.astype(int)):
                img[max(0, f-2):min(img_size, f+2), j] = 1.0
            labels.append(0)
        # Class 1: 차량 (전진) - Constant Doppler shift
        elif i % 3 == 1:
            img = np.zeros((img_size, img_size))
            start_freq = np.random.randint(10, img_size - 10)
            img[start_freq-3:start_freq+3, :] = 1.0
            labels.append(1)
        # Class 2: 드론 (호버링) - Multiple complex frequencies
        else:
            img = np.zeros((img_size, img_size))
            for _ in range(5):
                 t = np.linspace(0, np.pi * 2, img_size)
                 freq = np.sin(t * np.random.uniform(5, 10)) * (img_size / 8) + (img_size/2)
                 for j, f in enumerate(freq.astype(int)):
                    img[max(0, f):min(img_size, f+1), j] = np.random.rand()
            labels.append(2)

        images.append(img + np.random.rand(img_size, img_size) * 0.1) # Add noise
        
    return np.array(images).reshape(-1, img_size, img_size, 1), np.array(labels)

X_train, y_train = generate_spectrograms(300)
X_test, y_test = generate_spectrograms(60)
class_names = ['Person', 'Vehicle', 'Drone']

# 2. CNN 모델 설계
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax') # 3개 클래스 분류
])

# 3. 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 4. 결과 시각화
plt.figure(figsize=(10, 4))
for i in range(5):
    prediction = np.argmax(model.predict(X_test[i:i+1]))
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i].squeeze(), cmap='viridis')
    plt.title(f"Pred: {class_names[prediction]}\nTrue: {class_names[y_test[i]]}")
    plt.axis('off')
plt.show()