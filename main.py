from data_generation import *
from neuron import *
import matplotlib.pyplot as plt


def normalize(points):
    array = []
    max_x = max(point.x for point in points)
    min_x = min(point.x for point in points)
    max_y = max(point.y for point in points)
    min_y = min(point.y for point in points)
    for point in points:
        array.append(
            DataPoint(
                (point.x - min_x) / (max_x - min_x),
                (point.y - min_y) / (max_y - min_y),
                point.class_type

            )
        )

    return array


def create_plot(data, neuron):
    x_vals, y_vals, color_vals = zip(*[(float(i.x), float(i.y), i.class_type) for i in data])
    plt.scatter(x_vals, y_vals, c=color_vals)

    a, b = neuron.toLinear()
    y_line_vals = [(a * x + b) for x in x_vals]
    plt.plot(x_vals, y_line_vals)

    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.show()


def create_data(amount_of_points):
    array = []

    array += create_circle_of_points(int(amount_of_points / 2), 0, 70, 70, 25)
    array += create_circle_of_points(int(amount_of_points / 2), 1, 30, 30, 25)

    return array


def prepare_data(need_normalize):
    global min_lim, max_lim, data
    raw_data = create_data(200)
    print(raw_data)
    min_lim = 0
    max_lim = 100
    need_normalize = True
    if (need_normalize):
        max_lim = 1
        data = normalize(raw_data)
    else:
        data = raw_data
    return data


def HeavisideFunction(x):
    threshold = 0.0
    if x >= threshold:
        return 1
    else:
        return 0


if __name__ == '__main__':
    # np.random.seed(3)
    data = prepare_data(True)

    weights = np.random.uniform(0, 1, 2)
    weights = np.append(weights, 0)
    neuron = Neuron(weights, HeavisideFunction)
    train(neuron, data)
    create_plot(data, neuron)
