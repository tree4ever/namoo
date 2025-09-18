# Autoencoder: 레이더 신호 이상 탐지 (Anomaly Detection)
#   - 정상적인 레이더 신호(예: 안정적인 호흡 신호)만으로 Autoencoder를 학습시킵니다. 이후 비정상적인 신호(예: 갑작스러운 움직임, 간섭)가 들어왔을 때 
#       발생하는 높은 복원 오차(Reconstruction Error)를 이용해 이상 상황을 탐지
#   - 실습 주제: FMCW 레이더로 측정한 생체 신호(호흡)의 정상 패턴을 학습하고, 비정상적인 움직임이 발생했을 때를 탐지하는 Autoencoder 모델 개발.
#   - 데이터 생성 원리: 정상 신호는 사인파, 비정상 신호는 사인파 중간에 높은 진폭의 노이즈가 섞인 형태로 생성합니다.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. 가상 생체 신호 데이터 생성
def generate_vital_signs(n_samples=500, seq_length=200):
    # 정상 데이터 (학습용)
    normal_signals = []
    for _ in range(n_samples):
        freq = np.random.uniform(0.1, 0.4) # 호흡 주파수
        t = np.linspace(0, 2 * np.pi * 10, seq_length)
        normal_signals.append(np.sin(freq * t) + np.random.normal(0, 0.05, seq_length))
    
    # 비정상 데이터 (테스트용)
    anomaly_signals = []
    for _ in range(50):
        signal = normal_signals[_].copy()
        anomaly_point = np.random.randint(50, 150)
        signal[anomaly_point:anomaly_point+10] += np.random.uniform(2, 4, 10) # 갑작스러운 움직임
        anomaly_signals.append(signal)

    return np.array(normal_signals), np.array(anomaly_signals)

X_train_raw, X_anomaly_raw = generate_vital_signs()

# 2. 데이터 전처리 (개선 사항: MinMaxScaler 정규화)
# 모든 데이터를 0과 1 사이로 스케일링하여 학습 안정성 향상
scaler = MinMaxScaler()
X_train_flat = X_train_raw.reshape(-1, 1)
scaler.fit(X_train_flat)

X_train = scaler.transform(X_train_flat).reshape(X_train_raw.shape + (1,))
X_anomaly = scaler.transform(X_anomaly_raw.reshape(-1, 1)).reshape(X_anomaly_raw.shape + (1,))


# 3. LSTM Autoencoder 모델 설계
model = models.Sequential([
    # Encoder
    layers.Input(shape=(200, 1)),
    # 개선 사항: LSTM에서 불안정한 relu 활성화 함수 제거 (기본값인 tanh 사용)
    layers.LSTM(32, return_sequences=True),
    layers.LSTM(16, return_sequences=False),
    
    layers.RepeatVector(200),
    
    # Decoder
    layers.LSTM(16, return_sequences=True),
    layers.LSTM(32, return_sequences=True),
    
    # 개선 사항: 데이터가 0~1 사이이므로, 출력층에 sigmoid 활성화 함수 추가
    layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))
])
model.summary()

# 4. 모델 컴파일 및 학습 (정상 데이터만으로 학습!)
model.compile(optimizer='adam', loss='mae')
model.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.1)

# 5. 이상 탐지 결과 확인
reconstructed_normal = model.predict(X_train[:1])
reconstructed_anomaly = model.predict(X_anomaly[:1])

# 복원 오차 계산
mae_normal = np.mean(np.abs(X_train[:1] - reconstructed_normal), axis=1)
mae_anomaly = np.mean(np.abs(X_anomaly[:1] - reconstructed_anomaly), axis=1)

print("\n## 복원 오차 비교 ##")
print(f"정상 신호 복원 오차: {mae_normal[0][0]:.4f}")
print(f"이상 신호 복원 오차: {mae_anomaly[0][0]:.4f}")

# 6. 결과 시각화
plt.figure(figsize=(12, 5))

# Normal Signal Plot
plt.subplot(1, 2, 1)
plt.plot(X_train[0], label='Original Normal')
plt.plot(reconstructed_normal[0], label='Reconstructed')
plt.legend()
plt.title(f'Normal Signal (MAE: {mae_normal[0][0]:.4f})')
plt.ylim(0, 1)

# Anomaly Signal Plot
plt.subplot(1, 2, 2)
plt.plot(X_anomaly[0], label='Original Anomaly')
plt.plot(reconstructed_anomaly[0], label='Reconstructed')
plt.legend()
plt.title(f'Anomaly Signal (MAE: {mae_anomaly[0][0]:.4f})')
plt.ylim(0, 1)

plt.show()