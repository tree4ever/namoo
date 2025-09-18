#  Attention: 중요 구간에 집중하는 활동 분류
#   - LSTM만으로는 파악하기 어려운 복잡한 시퀀스에서, Attention 메커니즘을 추가하여 모델이 분류에 결정적인 구간에 더 '집중'하도록 만들어 성능을 높일 수 있음
#   - 실습 주제: 긴 노이즈 신호 속에서 짧고 희미한 ‘타겟’ 신호를 탐지하는 것
#   - 데이터 생성 원리: 배경 잡음에 가짜 방해 신호를 섞어 신호를 복잡하게 만들고, 절반의 데이터에만 고유한 패턴을 가진 ‘타겟’ 신호를 무작위 위치에 추가하여 Attention이 찾아내도록 유도

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt

# # 1. 가상 레이더 데이터 생성
# def generate_radar_echoes(n_samples=2000, seq_length=400, target_length=20):
#     sequences, labels = [], []
    
#     target_time = np.linspace(-3, 3, target_length)
#     target_signature = np.exp(-target_time**2) * 0.8
    
#     for _ in range(n_samples):
#         sequence = np.random.randn(seq_length) * 0.1
        
#         for _ in range(np.random.randint(1, 4)):
#             distractor_len = np.random.randint(5, 15)
#             distractor_amp = np.random.uniform(0.1, 0.3)
#             distractor_start = np.random.randint(0, seq_length - distractor_len)
#             sequence[distractor_start : distractor_start + distractor_len] += np.random.randn(distractor_len) * distractor_amp

#         if np.random.rand() > 0.5:
#             start_idx = np.random.randint(20, seq_length - target_length - 20)
#             sequence[start_idx : start_idx + target_length] += target_signature
#             labels.append(1)
#         else:
#             labels.append(0)
            
#         sequences.append(sequence)
        
#     return np.array(sequences).reshape(-1, seq_length, 1), np.array(labels)

# X_train, y_train = generate_radar_echoes(3000) # 훈련 데이터를 늘려 안정성 확보
# X_test, y_test = generate_radar_echoes(400)
# class_names = ['Normal Echo', 'Target Echo']

# # 2. Attention-LSTM 모델 설계
# inputs = layers.Input(shape=(400, 1))

# # LSTM
# lstm_out = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(inputs) 
# x = layers.Dropout(0.3)(lstm_out) 

# # Attention 레이어
# attention_result = layers.Attention(use_scale=True)([x, x])

# # 최종 분류
# x = layers.GlobalAveragePooling1D()(attention_result)
# x = layers.Dense(16, activation='relu')(x)
# outputs = layers.Dense(1, activation='sigmoid')(x)

# model = models.Model(inputs=inputs, outputs=outputs)

# # 3. 모델 컴파일 및 학습
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # Epochs를 늘려 세밀한 학습 유도
# model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# # 4. Attention 가중치 시각화를 위한 별도 모델 생성
# query = model.layers[1].output
# key = model.layers[1].output
# attention_scores = layers.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True))([query, key])
# attention_weights_raw = layers.Softmax(axis=-1)(attention_scores)
# visualization_model = models.Model(inputs=model.inputs, outputs=attention_weights_raw)

# # 5. 결과 확인 및 시각화
# target_sample_idx = np.where(y_test == 1)[0][0]
# sample = X_test[target_sample_idx:target_sample_idx+1]
# prediction = model.predict(sample)[0][0]
# predicted_class = class_names[1] if prediction > 0.5 else class_names[0]
# attention_weights = visualization_model.predict(sample)
# attention_weights = np.mean(attention_weights[0], axis=0)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
# fig.suptitle(f"Predicted: '{predicted_class}' (Prob: {prediction:.2f}) | True: '{class_names[1]}'", fontsize=16)
# ax1.plot(sample.squeeze(), label='Input Signal with Distractors')
# ax1.set_title("Radar Echo Signal ('Target' is the large pulse)")
# ax1.legend()
# ax2.plot(attention_weights, color='r', label='Attention Weight')
# ax2.set_title("Attention Mechanism Focus")
# ax2.set_xlabel("Time Step")
# ax2.legend()
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 가상 레이더 데이터 생성 (이전과 동일)
def generate_radar_echoes(n_samples=2000, seq_length=400, target_length=20):
    sequences, labels = [], []
    target_time = np.linspace(-3, 3, target_length)
    target_signature = np.exp(-target_time**2) * 1.0 # Target signature amplitude increased for clarity
    for _ in range(n_samples):
        sequence = np.random.randn(seq_length) * 0.1
        for _ in range(np.random.randint(2, 5)): # More distractors
            distractor_len = np.random.randint(5, 15)
            distractor_amp = np.random.uniform(0.1, 0.4)
            distractor_start = np.random.randint(0, seq_length - distractor_len)
            sequence[distractor_start : distractor_start + distractor_len] += np.random.randn(distractor_len) * distractor_amp
        if np.random.rand() > 0.5:
            start_idx = np.random.randint(0, seq_length - target_length)
            sequence[start_idx : start_idx + target_length] += target_signature
            labels.append(1)
        else:
            labels.append(0)
        sequences.append(sequence)
    return np.array(sequences).reshape(-1, seq_length, 1), np.array(labels)

X_train, y_train = generate_radar_echoes(4000) # Increased data samples
X_test, y_test = generate_radar_echoes(500)
class_names = ['Normal Echo', 'Target Echo']

# 2. Local Attention을 유도하는 CNN + Attention 모델
def build_final_model(input_shape=(400, 1)):
    inputs = layers.Input(shape=input_shape)

    # === 핵심 개선점: 계층적 CNN 특징 추출기 ===
    # Block 1: Broad feature detection
    x1 = layers.Conv1D(filters=16, kernel_size=11, padding='same', activation='relu')(inputs)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    
    # Block 2: Finer feature detection
    x2 = layers.Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')(x1)
    x2 = layers.Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling1D(pool_size=2)(x2)

    # Block 3: Even finer feature detection
    x3 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x2)
    x3 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x3)
    cnn_output = layers.MaxPooling1D(pool_size=2)(x3)
    # ==================================

    # Attention 레이어
    attention_result = layers.Attention(use_scale=True)([cnn_output, cnn_output])

    # 최종 분류
    x = layers.GlobalAveragePooling1D()(attention_result)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

model = build_final_model()
model.summary()

# 3. 모델 컴파일 및 학습
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

# 4. Attention 가중치 시각화 모델 생성
cnn_output_layer = model.layers[8].output # Last MaxPooling layer
query = cnn_output_layer
key = cnn_output_layer
attention_scores = layers.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True))([query, key])
attention_weights_raw = layers.Softmax(axis=-1)(attention_scores)
visualization_model = models.Model(inputs=model.inputs, outputs=attention_weights_raw)

# 5. 결과 확인 및 시각화
target_sample_idx = np.where(y_test == 1)[0][0]
sample = X_test[target_sample_idx:target_sample_idx+1]
prediction = model.predict(sample)[0][0]
predicted_class = class_names[1] if prediction > 0.5 else class_names[0]
attention_weights = visualization_model.predict(sample)
attention_weights = np.mean(attention_weights[0], axis=0) 
# CNN Max Pooling (2*2*2 = 8) 만큼 upsample
attention_weights_upsampled = np.repeat(attention_weights, 8) 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f"Predicted: '{predicted_class}' (Prob: {prediction:.2f}) | True: '{class_names[1]}'", fontsize=16)
ax1.plot(sample.squeeze(), label='Input Signal')
ax1.set_title("Radar Echo Signal")
ax1.legend()
ax2.plot(attention_weights_upsampled, color='r', label='Attention Weight')
ax2.set_title("Attention Mechanism Focus")
ax2.set_xlabel("Time Step")
ax2.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()