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


def trainNetForRegression(train_data: list[DataPoint], net: NeuronNet, error_level):
    speed = 0.01

    for layerIndex, layer in reversed(list(enumerate(net.layers))):
        for neuronIndex, neuron in layer:
            for point in train_data:
                expected = point.y
                prev_layer_output = net.processUpToLayer([point.x, 1], layerIndex)
                if layerIndex == len(net.layers):
                    # последний слой
                    y = neuron.process(prev_layer_output)
                    error = expected - y
                    delta = error * neuron.activation_func2(y)
                    neuron.delta = delta
                else:
                    sum_child_delta = 0
                    for child in net.layers[layerIndex + 1]:
                        sum_child_delta += child.delta * child.weihts[neuronIndex]
                    y = neuron.process(prev_layer_output)
                    delta = neuron.activation_func2(y) * sum_child_delta
                    neuron.delta = delta
    #TODO в итерации = обновление весов
    #TODO обновление выход из алгоритма обучения
    return net
