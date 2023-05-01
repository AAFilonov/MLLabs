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


def create_plot2(x_vals: list[float], y_vals: list[float], labels) -> None:
    plt.scatter(x_vals, y_vals, c=labels)
    plt.show()


def cubeFunction(x: float) -> float:
    return 1 * x * x - 100 * x


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


def define_model():
    model = tf.keras.Sequential([
        #layers.Dense(2, input_shape=[2], activation='tanh'),
        layers.Dense(1, input_shape=[2], activation=tf.keras.activations.sigmoid),
        layers.Dense(1, activation='softmax')
    ])
    return model


def createData(size:int):
    data = []
    data += create_circle_of_points(int(size / 2), 0, 30, 70, 25)
    data += create_circle_of_points(int(size / 2), 1, 70, 30, 25)
    x_vals, y_vals, color_vals = zip(
        *[(float(i.x), float(i.y), i.class_type) for i in data]
    )


    return x_vals, y_vals, color_vals


if __name__ == '__main__':



    x_train, y_train, label_train = createData(100)
    x_y_train_data = []
    for i in range(0, 100):
        arr = []
        arr.append(x_train[i])
        arr.append(y_train[i])
        x_y_train_data.append(arr)
    plt.scatter(x=x_train, y=y_train, c=label_train)
    plt.show()


    model = define_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['accuracy'],
                  run_eagerly=True)
    model.fit(x=x_y_train_data, y=label_train, epochs=100, batch_size=10)

    # С массивом Numpy
    data = np.random.random((100, 1))
    labels = np.random.random((100, 10))


    x_test, y_test, label_test = createData(100)
    x_y_test_data = []
    for i in range(0, 100):
        arr = []
        arr.append(x_train[i])
        arr.append(y_train[i])
        x_y_test_data.append(arr)

    t = model.predict(x_y_test_data)

    #plt.scatter(x=x_train, y=y_train, c=label_train)
    print(t)
