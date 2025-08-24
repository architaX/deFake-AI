# import tensorflow as tf
# import numpy as np
# import cv2

# # Load the trained model
# model = tf.keras.models.load_model('simple_deepfake_detector.h5')

# # Set image size (must match training)
# IMG_SIZE = (128, 128)

# def predict_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, IMG_SIZE)
#     img = img.astype('float32') / 255.0
#     img = np.expand_dims(img, axis=0)

#     prediction = model.predict(img)[0][0]

#     label = "REAL" if prediction >= 0.5 else "FAKE"
#     confidence = prediction if prediction >= 0.5 else 1 - prediction

#     print(f"{image_path} → Predicted: {label} (Confidence: {confidence:.2f})")

# # Test on some known examples
# predict_image('dataset/real/000045.jpg')
# predict_image('dataset/fake/f3.jpg')


# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_data = train_datagen.flow_from_directory(
#     'dataset/',
#     target_size=(128, 128),
#     batch_size=16,
#     class_mode='binary',
#     subset='training'
# )

# print(train_data.class_indices)




import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    'dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)
print(train_data.class_indices)


# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save model
model.save('simple_deepfake_detector.h5')
