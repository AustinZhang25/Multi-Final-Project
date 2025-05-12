import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from keras.api import layers, models
from keras.api.applications import ResNet50
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import labels from CSV
labels_data = pd.read_csv("data/echonest_processed.csv").values
print(f"Label shape: {labels_data.shape}")

def attach_label(image_path):
    image_id = int(image_path.split("/")[-1].split("_")[0])
    label = labels_data[labels_data[:, 0] == image_id, 1:].reshape(-1)
    if len(label) == 0:
        print(f"No label found for image {image_id}")
    return image_path, label

# Load image paths and labels
data_path = Path("spectrogram")
data_paths, labels = zip(*list(map(attach_label, sorted(map(str, data_path.glob("**/*.png"))))))

# Define image size for ResNet50
IMG_SIZE = (224, 224)

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((list(data_paths), list(labels)))
dataset = dataset.map(load_image).batch(8).prefetch(tf.data.AUTOTUNE)

# Split dataset into train and test
image_data = []
label_data = []

for img_batch, label_batch in tqdm(dataset):
    image_data.append(img_batch.numpy())
    label_data.append(label_batch.numpy())

X = np.concatenate(image_data, axis=0)
y = np.concatenate(label_data, axis=0)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Build ResNet50 model
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(8, activation='linear')  # Output for 8 regression targets
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train
with tf.device('/device:GPU:0'):
    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Predict
y_pred = model.predict(x_test)

# Save results
df = pd.DataFrame({
    **{f'y_test_{i}': y_test[:, i] for i in range(y_test.shape[1])},
    **{f'y_pred_{i}': y_pred[:, i] for i in range(y_pred.shape[1])}
})
df.to_csv("data/evaluate.csv", index=False)

# Report MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")
