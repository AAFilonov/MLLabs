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


def train(neuron, data):
    max_steps = 1000
    speed = 0.001
    results = []

    for i in range(0, max_steps):
        for point in data:
            expected = point.class_type
            neuron_result = neuron.process([point.x, point.y, 1])
            error = expected - neuron_result
            if error != 0:
                for k in range(0, 2):
                    neuron.weights[k] += speed * error * point.x
                results.append(error)
                print("error:" + str(error) + " neuron:" + str(neuron))

        if len(results) == 0:
            break
        else:
            results = []
    if len(results) == 0:
        print("Успех! Обучение закончилось обнулением ошибки")
    else:
        print("Провал! Обучение закончилось по таймауту")

    return neuron;
