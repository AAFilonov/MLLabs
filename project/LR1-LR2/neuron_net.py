from data_generation import *
from neuron import *


class NeuronNet:
    def __init__(self, inputCount):
        self.layers: list[list[Neuron]] = []
        self.inputCount = inputCount

    def addLayer(self, neuronCount: int, activationFunc: function):
        inputCountOfLayer = self.inputCount
        currentLayerCount = len(self.layers)
        if currentLayerCount != 0:
            inputCountOfLayer = len(self.layers[currentLayerCount])

        newLayer = []
        for i in range(0, neuronCount):
            newLayer.append(createNeuron(inputCountOfLayer, activationFunc))

        self.layers.append(newLayer)

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


def trainNet(train_data: list[DataPoint], net: NeuronNet):
    return net
