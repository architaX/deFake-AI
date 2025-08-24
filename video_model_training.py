# extracting frames from video

# import os
# import cv2
# import numpy as np
# from tqdm import tqdm

# INPUT_DIR = 'dataset_videos'
# OUTPUT_DIR = 'dataset_faces'
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# face_net = cv2.dnn.readNetFromCaffe(
#     "models/deploy.prototxt",
#     "models/res10_300x300_ssd_iter_140000.caffemodel"
# )

# def extract_faces_from_video(video_path, save_dir, label):
#     cap = cv2.VideoCapture(video_path)
#     frame_num = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         h, w = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
#         face_net.setInput(blob)
#         detections = face_net.forward()

#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.7:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 x1, y1, x2, y2 = box.astype("int")
#                 face = frame[y1:y2, x1:x2]
#                 if face.size == 0:
#                     continue
#                 face = cv2.resize(face, (128, 128))
#                 os.makedirs(save_dir, exist_ok=True)
#                 fname = f"{label}_{os.path.basename(video_path).split('.')[0]}_{frame_num}.jpg"
#                 cv2.imwrite(os.path.join(save_dir, fname), face)
#                 break
#         frame_num += 1
#     cap.release()

# for label in ['real', 'fake']:
#     input_folder = os.path.join(INPUT_DIR, label)
#     output_folder = os.path.join(OUTPUT_DIR, label)
#     os.makedirs(output_folder, exist_ok=True)
#     videos = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
#     for vid in tqdm(videos, desc=f"Processing {label} videos"):
#         extract_faces_from_video(os.path.join(input_folder, vid), output_folder, label)

# splitting dataset into train and test sets

# import os
# import shutil
# import random

# SOURCE = 'dataset_faces'
# DEST = 'data'
# SPLIT_RATIO = 0.8  # 80% train, 20% test

# for label in ['real', 'fake']:
#     files = os.listdir(os.path.join(SOURCE, label))
#     random.shuffle(files)
#     split = int(len(files) * SPLIT_RATIO)
#     train_files = files[:split]
#     test_files = files[split:]

#     for split_type, split_files in [('train', train_files), ('test', test_files)]:
#         out_dir = os.path.join(DEST, split_type, label)
#         os.makedirs(out_dir, exist_ok=True)
#         for f in split_files:
#             src = os.path.join(SOURCE, label, f)
#             dst = os.path.join(out_dir, f)
#             shutil.copy2(src, dst)


# training a deepfake detection model using TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=10, validation_data=test_data)

model.save('video_deepfake_detector.h5')
