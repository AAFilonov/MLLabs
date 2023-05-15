import os

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist

current_dir = os.getcwd()
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=current_dir + '/mnist.npz')

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)



test_model = keras.models.load_model('test_cnn_model')
my_model = keras.models.load_model('my_cnn_model')


test_score = test_model.evaluate(x_test, y_test_cat, verbose=0)

print("Test model score: %f" % test_score[0])
print("Test model accuracy: %f" % test_score[1])

my_score = test_model.evaluate(x_test, y_test_cat, verbose=0)

print("Test model score: %f" % my_score[0])
print("Test model accuracy: %f" % my_score[1])