# Attention: 중요 구간에 집중하는 활동 분류
#   - LSTM만으로는 파악하기 어려운 복잡한 시퀀스에서, Attention 메커니즘을 추가하여 모델이 분류에 결정적인 구간에 더 '집중'하도록 만들어 성능을 높일 수 있음
#   - 실습 주제: '접근-정지-접근'과 같이 복합적인 행동 시퀀스에서 '정지' 구간은 무시하고 '접근' 구간에 집중하여 
#     최종 행동을 '접근'으로 분류하는 Attention-LSTM 모델 개발
#   - 데이터 생성 원리: '접근' 신호(양수) 사이에 '정지' 신호(0)가 잠깐 포함된 복합 시퀀스를 생성

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 가상 복합 행동 시퀀스 데이터 생성
def generate_complex_sequence(n_samples=500, seq_length=150):
    sequences, labels = [], []
    for i in range(n_samples):
        if i % 2 == 0: # Class 0: 'Approach' (with a pause)
            seq1 = np.linspace(0.5, 1.0, 60)
            seq2 = np.random.normal(0, 0.1, 30) # Pause
            seq3 = np.linspace(1.0, 1.5, 60)
            seq = np.concatenate([seq1, seq2, seq3]) + np.random.normal(0, 0.1, seq_length)
            labels.append(0)
        else: # Class 1: 'Wandering' (random)
            seq = np.random.normal(0, 0.5, seq_length).cumsum() * 0.1
            labels.append(1)
        sequences.append(seq)
    return np.array(sequences).reshape(-1, seq_length, 1), np.array(labels)

X_train, y_train = generate_complex_sequence(1000)
X_test, y_test = generate_complex_sequence(100)
class_names = ['Approach', 'Wandering']

# 2. Attention을 포함한 LSTM 모델 설계
inputs = layers.Input(shape=(150, 1))
lstm_out = layers.LSTM(64, return_sequences=True)(inputs)

# Attention 레이어
# Query와 Value가 같을 때 Self-Attention으로 동작
query = lstm_out
value = lstm_out
attention_out = layers.Attention()([query, value])

# 최종 분류
x = layers.GlobalAveragePooling1D()(attention_out)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=inputs, outputs=outputs)

# 3. 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 4. 결과 확인
sample_idx = np.where(y_test == 0)[0][0] # 'Approach' 샘플
prediction = model.predict(X_test[sample_idx:sample_idx+1])[0][0]
plt.plot(X_test[sample_idx])
plt.title(f"Predicted: 'Approach' (Prob: {prediction:.2f})\nTrue: 'Approach'")
plt.show()