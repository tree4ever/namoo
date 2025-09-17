# YOLO의 컨셉을 응용한 CNN 기반 타겟 위치 추정
#   - YOLO 모델을 직접 훈련하는 것이 복잡하여, 그 핵심 컨셉인 '객체의 경계 상자(Bounding Box) 예측'을 CNN으로 간단히 구현해봄.
#     이는 이미지 내 타겟의 정확한 위치를 찾아내는 연구에 응용 가능
#   - 실습 주제: Range-Azimuth 이미지에 나타난 단일 타겟의 위치(x, y좌표, 너비, 높이)를 예측하는 CNN 회귀(Regression) 모델
#   - 데이터 생성 원리: 2D 이미지에 타겟 블록을 무작위로 생성하고, 해당 블록의 경계 상자 좌표 [x_center, y_center, width, height]를 정답 레이블로 사용

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. 가상 Range-Azimuth 이미지 및 Bounding Box 데이터 생성
def generate_localization_data(n_samples=500, img_size=64):
    images, boxes = [], []
    for _ in range(n_samples):
        image = np.zeros((img_size, img_size, 1))
        # 무작위 위치/크기로 타겟 생성
        w, h = np.random.randint(10, 20, 2)
        x, y = np.random.randint(0, img_size - w), np.random.randint(0, img_size - h)
        image[y:y+h, x:x+w, 0] = 1.0
        
        # Bounding Box 정규화 [x_center, y_center, width, height]
        box = np.array([
            (x + w / 2) / img_size,
            (y + h / 2) / img_size,
            w / img_size,
            h / img_size
        ], dtype=np.float32)
        
        images.append(image)
        boxes.append(box)
    return np.array(images), np.array(boxes)

X_train, y_train = generate_localization_data(2000)
X_test, y_test = generate_localization_data(200)

# 2. Bounding Box 예측을 위한 CNN 모델 설계
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='sigmoid') # 4개 좌표(0~1) 예측
])

# 3. 모델 컴파일 및 학습
# Bounding Box 회귀를 위한 손실 함수로 MSE 또는 MAE 사용
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 4. 위치 추정 결과 시각화
predicted_boxes = model.predict(X_test)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].squeeze(), cmap='gray')
    # 예측된 Box (빨간색)
    px, py, pw, ph = predicted_boxes[i] * 64
    pred_rect = patches.Rectangle((px - pw/2, py - ph/2), pw, ph, linewidth=2, edgecolor='r', facecolor='none', label='Predicted')
    ax.add_patch(pred_rect)
    # 실제 Box (녹색)
    tx, ty, tw, th = y_test[i] * 64
    true_rect = patches.Rectangle((tx - tw/2, ty - th/2), tw, th, linewidth=2, edgecolor='g', facecolor='none', label='True')
    ax.add_patch(true_rect)
    ax.axis('off')
plt.tight_layout()
plt.show()