import tensorflow as tf
from keras.applications.densenet import layers
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles

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




def linearFunction(x: int) -> int:
    return 0.05 * x * x - 4 * x + 100


def createData(random_state: int):
    #data = make_moons(noise=0.3, random_state=0),
    data = make_circles(noise=0.2, factor=0.5, random_state=random_state),

    x_vals= []
    y_vals = []
    color_vals = data[0][1]

    for pair in data[0][0]:
        x_vals.append(pair[0])
        y_vals.append(pair[1])

    return x_vals, y_vals, data[0][0], color_vals


def createData2(size: int):
    data = []
    data += create_circle_of_points(int(size / 4), 1, 30, 70, 20)
    data += create_circle_of_points(int(size / 4), 1, 70, 30, 20)
    data += create_circle_of_points(int(size / 4), 0, 30, 30, 20)
    data += create_circle_of_points(int(size / 4), 0, 70, 70, 20)
    x_vals, y_vals, color_vals = zip(
        *[(float(i.x), float(i.y), i.class_type) for i in data]
    )

    return x_vals, y_vals, color_vals

def define_model():
    model = tf.keras.Sequential([
        # layers.Dense(2, input_shape=[2], activation='tanh'),
        layers.Dense(10, input_shape=[2],  activation=tf.keras.activations.relu),
        layers.Dense(1, activation=tf.keras.activations.sigmoid),
    ])
    return model


if __name__ == '__main__':

    threshold = 0.5

    x_train, y_train, x_y_train_data, label_train = createData(1)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Horizontally stacked subplots')



    model = define_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=tf.keras.metrics.binary_crossentropy,
                  run_eagerly=True)
    model.fit(x=x_y_train_data, y=label_train, epochs=500, batch_size=100)
    train_result = model.predict(x_y_train_data)
    print(train_result)
    for i, r in enumerate(train_result):
        if r > threshold:
            train_result[i] = 1
        else:
            train_result[i] = 0

    axs[0, 0].scatter(x=x_train, y=y_train, c=label_train)
    axs[0, 0].set_title('Train labels')
    axs[0, 1].scatter(x=x_train, y=y_train, c=train_result)
    axs[0, 1].set_title('Train model prediction')
    # С массивом Numpy
    data = np.random.random((100, 1))
    labels = np.random.random((100, 10))

    x_test, y_test, x_y_test_data, label_test = createData(3)


    result = model.predict(x_y_test_data)
    print(result)
    for i, r in enumerate(result):
        if r > threshold:
            result[i] = 1
        else:
            result[i] = 0

    score = model.evaluate(x_y_test_data, )

    axs[1, 0].scatter(x=x_test, y=y_test, c=label_test)
    axs[1, 0].set_title('Test labels')
    axs[1, 1].scatter(x=x_test, y=y_test, c=result)
    axs[1, 1].set_title('Test model prediction')
    print(result)
    print(score)
    plt.show()
