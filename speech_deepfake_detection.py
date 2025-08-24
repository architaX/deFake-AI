import librosa
import numpy as np

def extract_mfcc(filepath):
    y, sr = librosa.load(filepath, sr=16000)  # Load with consistent sample rate
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfccs, axis=1)  # Average across time


# from tensorflow.keras import layers, models

# model = models.Sequential([
#     layers.Input(shape=(20, 1)),
#     layers.LSTM(64, return_sequences=True),
#     layers.LSTM(32),
#     layers.Dense(1, activation='sigmoid')  # Binary classification: Real vs Fake
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
