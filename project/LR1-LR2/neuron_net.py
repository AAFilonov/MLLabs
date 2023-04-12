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
            inputCountOfLayer = len(self.layers[currentLayerCount])

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
            new_layer_output = []
            for neuron in self.layers[i]:
                y = neuron.process(previous_layer_output)
                new_layer_output.append(y)

            previous_layer_output = new_layer_output.copy()
            if i == layerIndex:
                break
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


def trainNetForRegression(train_data: list[DataPoint], net: NeuronNet, error_level):
    speed = 0.01
    deltas: list[list[NeuronIterationData]] = []
    for layerIndex, layer in net.layers:
        for neuronIndex, neuron in layer:
            layer_deltas.append(0)

    for layerIndex, layer in reversed(list(enumerate(net.layers))):
        layer_deltas = []
        for neuronIndex, neuron in layer:
            delta = 0
            out = 0
            for point in train_data:
                expected = point.y
                prev_layer_output = net.processUpToLayer([point.x, 1], layerIndex)
                if layerIndex == len(net.layers):
                    # последний слой
                    out = neuron.process(prev_layer_output)
                    error = expected - y
                    delta = error * neuron.activation_func_derivative(y)
                else:
                    # не последний слой
                    sum_child_delta = 0
                    for childIndex, child in net.layers[layerIndex + 1]:
                        sum_child_delta += (
                            deltas[layerIndex + 1][childIndex][0]
                            * child.weihts[neuronIndex]
                        )
                    out = neuron.process(prev_layer_output)
                    delta = neuron.activation_func_derivative(y) * sum_child_delta
            layer_deltas.append((delta, out, prev_layer_output))
        deltas.append(layer_deltas)

    for layerIndex, layer in net.layers:
        for neuronIndex, neuron in layer:
            data = deltas[layerIndex][neuronIndex]
            delta = data[0]
            out = data[1]
            inputs = data[2]
            for weightIndex, weight in neuron.weights:
                weight = weight - speed*delta*out*inputs[weightIndex]

    # TODO в итерации = обновление весов
    # TODO обновление выход из алгоритма обучения
    return net
