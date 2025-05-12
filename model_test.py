import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pathlib


#split data into test(20%) and train(80%)
#data process
img_height, img_width = 180, 180

batch_size = 32

# Create the training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_fr  m_directory(
    data_directory,
    validation_split=0.8,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
base_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(img_height, img_width, 1),
                                                  pooling='avg',
                                                  weights='imagenet')
base_model.trainable = False

resnet_model = models.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(8, activation='linear')  # 8 features:
])







# Compile and train the model
resnet_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
resnet_model.summary()
resnet_model.fit(train_ds, validation_data=test_ds, epochs=20)

