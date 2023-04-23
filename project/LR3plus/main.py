import tensorflow as tf
from keras.applications.densenet import layers
import matplotlib.pyplot as plt

from data_generation import *


def create_plot(data: list[DataPoint], net) -> None:
    x_vals, y_vals, color_vals = zip(
        *[(float(i.x), float(i.y), i.class_type) for i in data]
    )
    plt.scatter(x_vals, y_vals, c=color_vals)

    if not net is None:
        x_line_vals = [(float(x) * 0.01) for x in range(1, 100, 1)]
        y_line_vals = [(net.process([x, 1])) for x in x_line_vals]

        plt.plot(x_line_vals, y_line_vals)

    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.show()


def cubeFunction(x: float) -> float:
    return 1*x * x - 100*x


def squareFunction(x: float) -> float:
    return x * x + 3 * x - 3


def create_data(amount_of_points: int) -> list[DataPoint]:
    array = []
    array += create_line_of_points(amount_of_points, 1, squareFunction)
    return array


def prepare_data(need_normalize: bool):
    global min_lim, max_lim, data
    raw_data = create_data(100)
    print(raw_data)
    min_lim = 0
    max_lim = 100
    if need_normalize:
        max_lim = 1
        data = normalize(raw_data)
    else:
        data = raw_data
    return data




if __name__ == '__main__':
    model = tf.keras.Sequential([
        # Добавляем полносвязный слой с 64 узлами к модели:
        layers.Dense(1, activation=tf.keras.activations.sigmoid),
        # Добавляем другой:
        layers.Dense(1, activation='softmax'),
        # Добавляем слой softmax с 10 выходами:
        layers.Dense(10, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    train_data = prepare_data(True)
    test_data = prepare_data(True)

    model.fit(data, labels, epochs=10, batch_size=32)


    create_plot(data, None)

