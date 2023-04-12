from data_generation import *


def createNeuron(inputsCount: int, actvationFunc, activationFuncPr):
    weights = np.random.uniform(0, 1, inputsCount)
    return Neuron(weights, actvationFunc, activationFuncPr)


class Neuron:
    def __init__(self, weights: list[float], activation_func,activation_func_derivative ):
        self.weights = weights
        self.inputsCount = len(weights)
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative


    def __str__(self):
        return "neuron: {" + str(self.weights) + "}"

    def __repr__(self):
        return str(self)

    def process(self, inputs: list[float]) -> float:
        if len(inputs) != len(self.weights):
            return None
        net = self.sum_func(inputs)
        return self.activation_func(net)

    def toLinear(self):
        a = -1 * self.weights[0] / self.weights[1]
        b = -1 * self.weights[2] / self.weights[1]
        return (a, b)

    def sum_func(self, inputs):
        net = 0
        for x, w in zip(inputs, self.weights):
            net += x * w
        return net


def trainNeuronForRegression(neuron: Neuron, data: list[DataPoint], error_level: float):
    max_steps = 1000
    speed = 0.001

    training = True
    for i in range(0, max_steps):
        error_quad_sum = 0
        for point in data:
            expected = point.y
            neuron_result = neuron.process([point.x, 1])
            error = expected - neuron_result

            if error != 0:
                #print("error:"+ str(error)+ " neuron:"+ str(neuron)+ " point:"+ str(point))
                error_quad_sum += error * error
                neuron.weights[0] += speed * error * point.x
                neuron.weights[1] += speed * error

        if error_quad_sum <= error_level:
            print(
                "Итерация "
                + str(i)
                + " закончилось достижением порога ошибки"
                + str(error_level)
                + ", обучение закончено"
            )
            training = False
            break
        else:
            print(
                "Итерация "
                + str(i)
                + " закончилось, сумма квадратов отклоений равна "
                + str(error_quad_sum)
            )
            error_quad_sum = 0

    if training:
        print("Провал! Обучение закончилось по таймауту")
    else:
        print(
            "Успех! Обучение закончилось достижением порога ошибки " + str(error_level)
        )
    print("Итоговое состояние нейрона: " + str(neuron))
    return neuron


def trainNeuronForClassififcation(neuron: Neuron, data: list[DataPoint]):
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
                print(
                    "error:"
                    + str(error)
                    + " neuron:"
                    + str(neuron)
                    + " point:"
                    + str(point)
                )
                error_count += 1
                neuron.weights[0] += speed * error * point.x
                neuron.weights[1] += speed * error * point.y
                neuron.weights[2] += speed * error

        if error_count == 0:
            print(
                "Итерация "
                + str(i)
                + " закончилось обнулением ошибки, обучение закончено"
            )
            training = False
            break
        else:
            print(
                "Итерация "
                + str(i)
                + " закончилось, количество ошибок равно "
                + str(error_count)
            )
            error_count = 0

    if training:
        print("Провал! Обучение закончилось по таймауту")
    else:
        print("Успех! Обучение закончилось обнулением ошибки")
    print("Итоговое состояние нейрона: " + str(neuron) + str(neuron.toLinear()))
    return neuron
