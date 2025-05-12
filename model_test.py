import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import mean_squared_error
from tensorflow.keras.backend import clear_session

# Clear any previous session
clear_session()
# Import labels
labels_data = pd.read_csv('data/echonest_norm.csv').values
print(f"Label shape: {labels_data.shape}")

def attach_label(image_path):
    try:
        image_id = int(image_path.split("/")[-1].split("_")[0])
        label = labels_data[labels_data[:, 0] == image_id, 1:]
        if label.shape[0] == 0:
            print(f"No label found for image {image_id}")
            return None
        return image_path, label.reshape(-1)
    except Exception as e:
        print(f"Error with {image_path}: {e}")
        return None

# Load image paths and labels
data_path = Path("spectrogram")
all_image_paths = sorted(map(str, data_path.glob("**/*.png")))
valid_pairs = list(filter(None, map(attach_label, all_image_paths)))

if len(valid_pairs) == 0:
    raise ValueError("No valid image-label pairs found!")

data_paths, labels = zip(*valid_pairs)

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (int(984 / 3), int(2385 / 3)))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((list(data_paths), list(labels)))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and split dataset
train_size = int(0.8 * len(data_paths))
train_dataset = dataset.take(train_size).batch(4).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).batch(4).prefetch(tf.data.AUTOTUNE)

print("Dataset prepared.")

# Build model
clear_session()
base_model = ResNet50(weights=None, include_top=False, input_shape=(int(984 / 3), int(2385 / 3), 3))
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(8, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

print("Start training")
try:
    with tf.device('/device:GPU:0'):
        model.fit(train_dataset, epochs=20, validation_data=val_dataset)
except Exception as e:
    print(f"Training failed: {e}")
    exit(1)

# Save results
y_test = []
y_pred = []
for images, batch_labels in tqdm(val_dataset):
    y_test.append(batch_labels.numpy())
    y_pred.append(model.predict(images))
y_test = np.concatenate(y_test, axis=0)
y_pred = np.concatenate(y_pred, axis=0)

df = pd.DataFrame({
    **{f'y_test_{i}': y_test[:, i] for i in range(y_test.shape[1])},
    **{f'y_pred_{i}': y_pred[:, i] for i in range(y_pred.shape[1])}
})
df.to_csv("data/evaluate.csv", index=False)

# Report MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")
