from data_generation import *


class Neuron:
    def __init__(self, weights, activation_func):
        self.weights = weights
        self.activation_func = activation_func

    def __str__(self):
        return "neuron: {" + str(self.weights) + "}"

    def __repr__(self):
        return str(self)

    def process(self, inputs):
        if len(inputs) != len(self.weights):
            return None
        net = self.sum_func(inputs)
        return self.activation_func(net)

    def toLinear(self):
        a = self.weights[0] / self.weights[1]
        b = self.weights[2] / self.weights[1]
        return (a, b)

    def sum_func(self, inputs):
        net = 0
        for x, w in zip(inputs, self.weights):
            net += x * w
        return net


def train(neuron: Neuron, data: list[DataPoint]):
    max_steps = 1000
    speed = 0.001

    training = True
    for i in range(0, max_steps):
        error_count = 0
        for point in data:
            expected = point.class_type
            neuron_result = neuron.process([point.x, point.y, 1])
            error = expected - neuron_result

            if error != 0:
                print("error:" + str(error) + " neuron:" + str(neuron) + " point:" + str(point))
                error_count += 1
                neuron.weights[0] += speed * error * point.x
                neuron.weights[1] += speed * error * point.y
                neuron.weights[2] += speed * error

        if error_count == 0:
            print("Итерация " + str(i) + " закончилось обнулением ошибки, обучение закончено")
            training = False
            break
        else:
            print("Итерация " + str(i) + " закончилось, количество ошибок равно " + str(error_count))
            error_count = 0

    if training:
        print("Провал! Обучение закончилось по таймауту")
    else:
        print("Успех! Обучение закончилось обнулением ошибки")
    print("Итоговое состояние нейрона: " + str(neuron) + str(neuron.toLinear()))
    return neuron
