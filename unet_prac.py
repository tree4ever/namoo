# U-Net: Range-Doppler Map에서 타겟 영역 분할 (Segmentation)
#   - U-Net은 이미지의 픽셀 단위 분할에 매우 강력합니다. 이를 이용해 Range-Doppler Map(RDM)이나 SAR 이미지에서 노이즈/클러터와 타겟 신호 영역을 정밀하게 분리
#   - 실습 주제: Range-Doppler Map 이미지에서 클러터(배경 잡음)로부터 타겟(차량, 사람 등)의 신호 영역을 정확히 분리해내는 U-Net 모델
#   - 데이터 생성 원리: 2D 배열에 무작위 배경 잡음을 생성하고, 특정 위치에 더 밝은 값을 갖는 타겟 '블록(blob)'을 추가하여 RDM을 모방합니다. 
#       정답 데이터(Mask)는 타겟 영역만 1로 표시

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 가상 Range-Doppler Map 및 Mask 데이터 생성
def generate_rdm_data(n_samples=50, img_size=128):
    images, masks = [], []
    for _ in range(n_samples):
        # 배경 클러터 생성
        image = np.random.rand(img_size, img_size) * 0.3
        mask = np.zeros((img_size, img_size))

        # 무작위 위치에 타겟 생성
        x, y = np.random.randint(20, img_size - 20, 2)
        h, w = np.random.randint(10, 20, 2)
        image[y:y+h, x:x+w] += np.random.rand(h, w) * 0.7 + 0.3
        mask[y:y+h, x:x+w] = 1.0
        
        images.append(image)
        masks.append(mask)
        
    return np.array(images).reshape(-1, img_size, img_size, 1), np.array(masks).reshape(-1, img_size, img_size, 1)

X_train, y_train = generate_rdm_data(400)
X_test, y_test = generate_rdm_data(50)

# 2. U-Net 모델 설계
def build_unet_model(input_shape):
    inputs = layers.Input(input_shape)
    # 인코더 (Contracting Path)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    # 병목
    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    # 디코더 (Expansive Path) with Skip Connections
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    u2 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)
    return models.Model(inputs, outputs)

model = build_unet_model((128, 128, 1))

# 3. 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

# 4. 분할 결과 시각화
predicted_masks = model.predict(X_test)

plt.figure(figsize=(12, 8))
for i in range(3):
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(X_test[i].squeeze(), cmap='viridis')
    plt.title('Input RDM')
    plt.axis('off')

    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(y_test[i].squeeze(), cmap='gray')
    plt.title('True Mask')
    plt.axis('off')

    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(predicted_masks[i].squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
plt.tight_layout()
plt.show()