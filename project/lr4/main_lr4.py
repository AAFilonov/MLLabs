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
last_model = keras.models.load_model('last_cnn_model')

test_score = test_model.evaluate(x_test, y_test_cat, verbose=0)
my_score = my_model.evaluate(x_test, y_test_cat, verbose=0)
last_score = last_model.evaluate(x_test, y_test_cat, verbose=0)

print("Test model score:    %f" % test_score[0])
print("Test model accuracy: %f%%" % (test_score[1] * 100))
print("Test model errors:   %f%%" % ((1 - test_score[1]) * 100))

print("My   model score:    %f" % my_score[0])
print("My   model accuracy: %f%%" % (my_score[1] * 100))
print("My   model errors:   %f%%" % ((1 - my_score[1]) * 100))

print("Score decrease:      %f " % (my_score[0] - test_score[0]))
print("Accuracy growth:     %f%%" % (((my_score[1] - test_score[1])) * 100))
#print("Errors decrease:     %f%%" % ((((1 - test_score[1]) / (1 - my_score[1]))) * 100))

print(" ")
print("Last  model score:    %f" % last_score[0])
print("Last  model accuracy: %f%%" % (last_score[1] * 100))
print("Last  model errors:   %f%%" % ((1 - last_score[1]) * 100))


print("Score decrease:      %f " % (my_score[0] - last_score[0]))
print("Accuracy growth:     %f%%" % (((my_score[1] - last_score[1])) * 100))

my_model.save('last_cnn_model')
