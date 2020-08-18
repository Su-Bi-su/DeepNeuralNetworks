# imports

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Normialize the input data

x_train, x_test = x_train / 255.0, x_test / 255.0


# Sequential API (very convenient but not very flexible) - it only allows you to one input map to one output
model = keras.models.Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512, activation='relu'),        # for fully connected layer
        layers.Dense(256, activation='relu'),
        layers.Dense(10)                             # last layer so we don't need specify activation function. we will be using softmax which will be included with cross entropy loss function

    ]
)

print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Network is not running in GPU. Please install GPU version of TF")

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
