# LSTM: 시계열 레이더 신호 기반 활동 분류
#   - 움직이는 타겟의 시간에 따른 레이더 신호(예: 도플러 변화) 시퀀스를 LSTM에 입력하여 어떤 활동을 하는지 분류
#   - 실습 주제: 시간에 따라 변하는 1D 레이더 신호 시퀀스를 분석하여 '정지', '접근', '후퇴' 상태 분류
#   - 데이터 생성 원리: 정지는 0에 가까운 값, 접근은 양수, 후퇴는 음수 값을 갖는 시계열 신호를 생성

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 가상 시계열 레이더 신호 데이터 생성
def generate_time_series_data(n_samples=100, seq_length=100):
    sequences, labels = [], []
    for i in range(n_samples):
        # Class 0: 정지
        if i % 3 == 0:
            seq = np.random.normal(0, 0.1, seq_length)
            labels.append(0)
        # Class 1: 접근
        elif i % 3 == 1:
            seq = np.linspace(0.5, 1.5, seq_length) + np.random.normal(0, 0.2, seq_length)
            labels.append(1)
        # Class 2: 후퇴
        else:
            seq = np.linspace(-0.5, -1.5, seq_length) + np.random.normal(0, 0.2, seq_length)
            labels.append(2)
        sequences.append(seq)
        
    return np.array(sequences).reshape(-1, seq_length, 1), np.array(labels)

X_train, y_train = generate_time_series_data(300)
X_test, y_test = generate_time_series_data(60)
class_names = ['Stop', 'Approach', 'Recede']

# 2. LSTM 모델 설계
model = models.Sequential([
    layers.Input(shape=(100, 1)),
    layers.LSTM(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# 3. 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 4. 결과 시각화
plt.figure(figsize=(12, 5))
for i in range(3):
    sample_idx = np.where(y_test == i)[0][0] # 각 클래스별 샘플 하나씩 선택
    prediction = np.argmax(model.predict(X_test[sample_idx:sample_idx+1]))
    plt.subplot(1, 3, i+1)
    plt.plot(X_test[sample_idx])
    plt.title(f"Pred: {class_names[prediction]}\nTrue: {class_names[y_test[sample_idx]]}")
plt.tight_layout()
plt.show()