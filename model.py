import os

from lazy_loader import attach
from tqdm import tqdm
import keras.api.applications.resnet50
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.api.utils import image_dataset_from_directory
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Import labels from csv
labels_data = pd.read_csv("data/echonest_processed.csv").values
print(f"Label shape: {labels_data.shape}")

def attach_label(image_path):
    image_id = int(image_path.split("/")[-1].split("_")[0])
    label = labels_data[labels_data[:, 0] == image_id, 1:].reshape(-1)
    if len(label) == 0:
        print(f"No label found for image {image_id}")
    return image_path, label

# Paths
data_path = Path("spectrogram")
data_paths, labels = zip(*list(map(attach_label, sorted(map(str, data_path.glob("**/*.png"))))))

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (int(984 / 3), int(2385 / 3)))
    image = keras.api.applications.resnet50.preprocess_input(image)
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((list(data_paths), list(labels))).map(load_image).batch(4)

resnet = keras.api.applications.resnet50.ResNet50(include_top=False, input_shape=(int(984 / 3), int(2385 / 3), 3), pooling='avg')

# Extract features in batches
features = []
labels = []
for images, batch_labels in tqdm(dataset):
    batch_features = resnet(images, training=False)
    features.append(batch_features.numpy())
    labels.append(batch_labels.numpy())

# Combine all features and labels
x = np.concatenate(features, axis=0)
y = np.concatenate(labels, axis=0)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Train a linear regression model
model = Ridge(alpha=0.1)
with tf.device('/device:GPU:0'):
    model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)

# Prepare csv
columns = {}
for i in range(y_test.shape[1]):
    columns[f'y_test_{i}'] = y_test[:, i]
    columns[f'y_pred_{i}'] = y_pred[:, i]
df = pd.DataFrame(columns)
df.to_csv("data/evaluate.csv", index=False)

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")
