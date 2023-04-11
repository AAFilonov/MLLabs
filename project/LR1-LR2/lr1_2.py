from data_generation import *
from neuron import *
import matplotlib.pyplot as plt


def create_plot(data: list[DataPoint], neuron: Neuron) -> None:
    x_vals, y_vals, color_vals = zip(
        *[(float(i.x), float(i.y), i.class_type) for i in data]
    )
    plt.scatter(x_vals, y_vals, c=color_vals)

    a = neuron.weights[0]
    b = neuron.weights[1]
    x_line_vals = [(float(x) * 0.01) for x in range(1, 100, 1)]
    y_line_vals = [(a * x + b) for x in x_line_vals]

    plt.plot(x_line_vals, y_line_vals)

    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.show()


def linearFunction(x: int) -> int:
    return -2 * x + 3


def create_data(amount_of_points: int) -> list[DataPoint]:
    array = []

    array += create_line_of_points(amount_of_points, 1, linearFunction)

    return array


def prepare_data(need_normalize: bool):
    global min_lim, max_lim, data
    raw_data = create_data(200)
    print(raw_data)
    min_lim = 0
    max_lim = 100
    if need_normalize:
        max_lim = 1
        data = normalize(raw_data)
    else:
        data = raw_data
    return data


def linearFunc(x: int):
    return x


if __name__ == "__main__":
    # np.random.seed(3)
    data = prepare_data(True)

    weights = np.random.uniform(0, 1, 2)
    neuron = Neuron(weights, linearFunc)
    trainNeuronForRegression(neuron, data, 0.5)
    create_plot(data, neuron)
