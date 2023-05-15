import os

import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow import keras

current_dir = os.getcwd()
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=current_dir + '/mnist.npz')

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

model = keras.Sequential([

    Conv2D(filters=16,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu',
           input_shape=(28, 28, 1)),
    Conv2D(filters=16,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu',
           input_shape=(28, 28, 1)),

    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)),
    Conv2D(filters=32,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu',
           input_shape=(28, 28, 1)),
    Conv2D(filters=32,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu',
           input_shape=(28, 28, 1)),
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)),
    Conv2D(filters=64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu',
           input_shape=(28, 28, 1)),
    Conv2D(filters=64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu',
           input_shape=(28, 28, 1)),
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)),

    Conv2D(filters=128,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu',
           input_shape=(28, 28, 1)),
    Conv2D(filters=128,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu',
           input_shape=(28, 28, 1)),
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)),

    Flatten(),

    # тут могут быть скрытые полносвязные слои с функцией активации ReLU

    # Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train_cat,
          batch_size=32,
          epochs=1,
          verbose=1)

score = model.evaluate(x_train, y_train_cat, verbose=0)

print("Train score: %f" % score[0])
print("Train accuracy: %f" % score[1])

score = model.evaluate(x_test, y_test_cat, verbose=0)

print("Test score: %f" % score[0])
print("Test accuracy: %f" % score[1])

model.save('my_cnn_model')
