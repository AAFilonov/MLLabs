from data_generation import *
from neuron import *
from neuron_net import *
import matplotlib.pyplot as plt


def create_plot(data: list[DataPoint], net) -> None:
    x_vals, y_vals, color_vals = zip(
        *[(float(i.x), float(i.y), i.class_type) for i in data]
    )
    plt.scatter(x_vals, y_vals, c=color_vals)

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


def sigmoidFunc(x: int):
    return 1 / (1 + math.exp(-x))


def sigmoidFuncPR(x: int):
    return sigmoidFunc(x) * (1 - sigmoidFunc(x))


def reLU2Func(x: int):
    return max(0.01 * x, x)


def reLU2FuncPr(x: int):
    if x < 0:
        return 0.01
    else:
        return 1


def summrFunc(x: int):
    return x


def summFuncPr(x: int):
    return 1


def main_square():
    np.random.seed(3)
    data = prepare_data(True)
    net = NeuronNet(2)
    net.addLayer(1, sigmoidFunc, sigmoidFuncPR)
    net.addLayer(1, summrFunc, summFuncPr)

    trainNetForRegression(data, net, 0.001, 4000, 0.3)
    create_plot(data, net)


def main_cube():
    data = prepare_data(True)
    net = NeuronNet(2)
    net.addLayer(4, sigmoidFunc, sigmoidFuncPR)
    net.addLayer(1, summrFunc, summFuncPr)

    trainNetForRegression(data, net, 0.001, 4000, 0.2)
    create_plot(data, net)


if __name__ == "__main__":
    main_square()
    #main_cube()
