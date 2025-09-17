# MLP: 추출된 특징(Feature) 기반 타겟 분류
#   - 레이더 신호에서 직접 추출한 물리적 특징(RCS, 속도, SNR 등)들을 입력 벡터로 만들어 MLP로 분류하는 것은 가장 기본적이면서도 효과적인 접근 방식
#   - 실습 주제: 타겟의 RCS(크기), 평균 도플러(속도), SNR(신호 품질) 특징 벡터를 기반으로 타겟을 '드론', '새', '사람'으로 분류하는 MLP 모델
#   - 데이터 생성 원리: 각 클래스별로 특징 값의 분포가 다르도록 가상 데이터를 생성 
#                   (예: 드론은 RCS가 작고 속도가 빠름, 사람은 RCS가 중간이고 속도가 느림)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import seaborn as sns

# 1. 가상 특징 벡터 데이터 생성
def generate_feature_data(n_samples=500):
    features, labels = [], []
    for i in range(n_samples):
        # Class 0: 드론 (RCS 작음, 속도 빠름, SNR 중간)
        if i % 3 == 0:
            rcs = np.random.normal(1, 0.5)
            doppler = np.random.normal(15, 3)
            snr = np.random.normal(15, 2)
            labels.append(0)
        # Class 1: 새 (RCS 매우 작음, 속도 중간)
        elif i % 3 == 1:
            rcs = np.random.normal(0.5, 0.2)
            doppler = np.random.normal(8, 2)
            snr = np.random.normal(10, 2)
            labels.append(1)
        # Class 2: 사람 (RCS 큼, 속도 느림)
        else:
            rcs = np.random.normal(5, 1)
            doppler = np.random.normal(1.5, 0.5)
            snr = np.random.normal(20, 3)
            labels.append(2)
        features.append([rcs, doppler, snr])
    return np.array(features), np.array(labels)

X_train, y_train = generate_feature_data(1000)
X_test, y_test = generate_feature_data(150)
class_names = ['Drone', 'Bird', 'Person']

# 데이터 분포 시각화 (EDA)
df = pd.DataFrame(X_train, columns=['RCS', 'Doppler', 'SNR'])
df['label'] = [class_names[l] for l in y_train]
sns.pairplot(df, hue='label')
plt.show()

# 2. 데이터 전처리 (정규화)
scaler = tf.keras.layers.Normalization(axis=-1)
scaler.adapt(X_train)
X_train_scaled = scaler(X_train)
X_test_scaled = scaler(X_test)

# 3. MLP 모델 설계
model = models.Sequential([
    layers.Input(shape=(3,)), # 3개의 특징 (RCS, Doppler, SNR)
    scaler, # 정규화 층
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# 4. 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))