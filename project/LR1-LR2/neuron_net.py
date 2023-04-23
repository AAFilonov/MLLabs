from data_generation import *
from neuron import *


class NeuronNet:
    def __init__(self, inputCount):
        self.layers: list[list[Neuron]] = []
        self.inputCount = inputCount

    def addLayer(self, neuronCount: int, activationFunc, activationFuncPr):
        inputCountOfLayer = self.inputCount
        currentLayerCount = len(self.layers)
        if currentLayerCount != 0:
            inputCountOfLayer = len(self.layers[currentLayerCount - 1])

        newLayer = []
        for i in range(0, neuronCount):
            newLayer.append(
                createNeuron(inputCountOfLayer, activationFunc, activationFuncPr)
            )

        self.layers.append(newLayer)

    def processUpToLayer(self, inputs: list[float], layerIndex: int) -> list[float]:
        if len(inputs) != self.inputCount:
            return None

        previous_layer_output = inputs
        for i in range(0, len(self.layers)):
            if i == layerIndex:
                break
            new_layer_output = []
            for neuron in self.layers[i]:
                y = neuron.process(previous_layer_output)
                new_layer_output.append(y)

            previous_layer_output = new_layer_output.copy()

        # выход последнего ОБРАБОТАННОГО слоя будет в previous_layer_output
        return previous_layer_output

    def process(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.inputCount:
            return None

        previous_layer_output = inputs
        for layer in self.layers:
            new_layer_output = []
            for neuron in layer:
                y = neuron.process(previous_layer_output)
                new_layer_output.append(y)

            previous_layer_output = new_layer_output.copy()

        # выход последнего слоя будет в previous_layer_output
        return previous_layer_output


def trainNeuron(
    neuron: Neuron,
    layerIndex: int,
    train_data: list[DataPoint],
    net: NeuronNet,
    error_level: float,
) -> None:
    return


NeuronIterationData = tuple[int, int, list[int]]


def trainNetForRegression(
    train_data: list[DataPoint], net: NeuronNet, error_level , max_iterarions
) -> NeuronNet:
    previous_error = 10000000000000
    for i in range(0, max_iterarions):
        net, error = do_iteration(train_data, net)
        if error <= error_level:
            print(
                "Успех, алгоритм закончился достижемнием порога "
                + str(error_level)
                + " на итерации "
                + str(i)
            )
            return net

        else:
            if False and error >= previous_error:
                print(
                    "Остановка. Итерация "
                    + str(i)
                    + ", уровень ошибки начал расти с отметки"
                    + str(error)
                )
                break
            else:
                print("Итерация " + str(i) + ", уровень ошибки равен " + str(error))
                previous_error = error

    print(
        "Алгоритм закончился достижемнием порога числа итераций " + str(max_iterarions)
    )
    return net


def do_iteration(
    train_data: list[DataPoint], net: NeuronNet
) -> tuple[NeuronNet, float]:
    speed = 0.001
    iteration_error = 0
    deltas: list[list[NeuronIterationData]] = []
    for layerIndex in range(0, len(net.layers)):
        layer = net.layers[layerIndex]
        deltas.append([])
        for neuronIndex in range(0, len(layer)):
            neuron = layer[neuronIndex]
            deltas[layerIndex].append((0, None, None))

    for point in train_data:
        for layerIndex, layer in reversed(list(enumerate(net.layers))):
            layer_deltas = []
            for neuronIndex, neuron in enumerate(layer):
                delta = 0
                out = 0
                expected = point.y
                prev_layer_output = net.processUpToLayer([point.x, 1], layerIndex)
                if layerIndex == len(net.layers) - 1:
                    # последний слой
                    out = neuron.process(prev_layer_output)
                    error = out - expected
                    delta = error * neuron.activation_func_derivative(out)
                    iteration_error += error * error
                else:
                    # не последний слой
                    sum_child_delta = 0
                    for childIndex, child in enumerate(net.layers[layerIndex + 1]):
                        sum_child_delta += (
                            deltas[layerIndex + 1][childIndex][0]
                            * child.weights[neuronIndex]
                        )
                    out = neuron.process(prev_layer_output)
                    delta = neuron.activation_func_derivative(out) * sum_child_delta
                deltas[layerIndex][neuronIndex] = (delta, out, prev_layer_output)
            # layer_deltas.append((delta, out, prev_layer_output))
        for layerIndex, layer in enumerate(net.layers):
            for neuronIndex, neuron in enumerate(layer):
                data = deltas[layerIndex][neuronIndex]
                delta = data[0]
                out = data[1]
                inputs = data[2]
                for weightIndex, weight in enumerate(neuron.weights):
                    neuron.weights[weightIndex] = (
                        weight - speed * delta * out * inputs[weightIndex]
                    )   
        # deltas.append(layer_deltas)

   

    # TODO обновление выход из алгоритма обучения
    return (net, iteration_error)
