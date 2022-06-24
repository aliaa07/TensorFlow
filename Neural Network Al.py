import time
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


def timer(func):
    def timeit(*args):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        print("time measured by timer function: ", (end - start) * 10000)

    return timeit


def pprint(dictionary):
    print("***********************************")
    for key, val in dictionary.items():
        print(key)
        print(val)
    print("***********************************")


class NeuralNetwork:
    Examples = Classes = Features = 0
    Layers = list()
    Matrix_Weight = dict()
    Matrix_ProductSum = dict()
    Matrix_Output = dict()
    Matrix_delta = dict()
    Matrix_DELTA = dict()

    def __init__(self, X: np.ndarray = None, Y: np.ndarray = None):
        self.X = X
        self.Y = Y
        if self.X.ndim == self.Y.ndim and self.X.shape[0] == self.Y.shape[0]:
            if self.X.ndim == 1:
                print("*** Note: we have increased the dims of X by one in axis=-1 ***\n")
                self.X = np.expand_dims(self.X, axis=-1)
                self.Y = np.expand_dims(self.Y, axis=-1)
        else:
            raise ValueError("Number of input and output examples dose not match")
        self.Examples = self.X.shape[0]
        self.Features = self.X.shape[1]
        self.Classes = self.Y.shape[1]
        self.Layers.append(self.Features)  # input neurons (initializers)
        self.Layers.append(self.Classes)  # output neurons (classes)

    def add_layer(self, *neurons: int, list_of_layers: object = None) -> None:
        if list_of_layers is not None:
            neurons = list_of_layers
        for neuron in neurons:
            self.Layers.insert(-1, neuron)

    def details(self):
        print(self.Layers)
        print(self.Classes)
        print(self.Examples)
        print(self.Features)

    def Weight_init(self) -> None:
        for index, neuron in enumerate(self.Layers):
            if index == 0:
                continue
            self.Matrix_Weight[f"W{index}"] = np.random.random([neuron, self.Layers[index - 1] + 1])

    def ProductSum_init(self) -> None:
        self.Matrix_ProductSum.clear()
        for index, neuron in enumerate(self.Layers):
            if index == 0:
                continue
            self.Matrix_ProductSum[f"H{index}"] = np.zeros([1, neuron])

    def Output_init(self, Xi: np.ndarray or int) -> None:
        self.Matrix_Output.clear()
        for index, neuron in enumerate(self.Layers):
            self.Matrix_Output[f"O{index}"] = np.zeros([1, neuron])
            if index == 0:
                if isinstance(Xi, int):
                    Xi = np.array([Xi])
                self.Matrix_Output[f"O{index}"] = np.expand_dims(Xi, axis=0)

    @staticmethod
    def Adding_Bias(Matrix):
        return np.expand_dims(np.insert(Matrix, 0, 1), axis=0)

    @staticmethod
    def Sigmoid(z):
        return (1 + np.exp(-z)) ** -1

    @staticmethod
    def Liner(z):
        return z

    def Forward_Propagation_ProductSum(self, Xi: np.ndarray or int, activation):
        self.ProductSum_init()
        self.Output_init(Xi)
        for index in range(1, len(self.Layers)):
            self.Matrix_ProductSum[f"H{index}"] = np.dot(self.Matrix_Weight[f"W{index}"],
                                                         np.transpose(self.Adding_Bias(self.Matrix_Output[f"O{index - 1}"])))
            if index != len(self.Layers) - 1:
                self.Matrix_Output[f"O{index}"] = activation(self.Matrix_ProductSum[f"H{index}"])

    def Forward_Propagation_Output(self, activation):
        for index in range(1, len(self.Layers) - 1):
            self.Matrix_Output[f"O{index}"] = activation(self.Matrix_ProductSum[f"H{index}"])

    def Forward_Propagation_Output_Final_Layer(self, activation):
        self.Matrix_Output[f"O{len(self.Layers) - 1}"] = activation(self.Matrix_ProductSum[f"H{len(self.Layers) - 1}"])

    def Forward_Propagation(self, Xi: np.ndarray or int, activation_hidden_layer, activation_final_layer):
        self.Forward_Propagation_ProductSum(Xi, activation_hidden_layer)
        self.Forward_Propagation_Output_Final_Layer(activation_final_layer)
        return self.Matrix_Output[f"O{len(self.Layers) - 1}"]

    def delta_init(self, Yi: np.ndarray or int) -> None:
        self.Matrix_delta.clear()
        for index, neuron in enumerate(self.Layers):
            if index == 0:
                continue
            self.Matrix_delta[f"d{index}"] = np.zeros([neuron, 1])
            if index == len(self.Layers) - 1:
                if isinstance(Yi, int):
                    Yi = np.array([Yi])
                self.Matrix_delta[f"d{index}"] = np.expand_dims(Yi, axis=-1)

    def DELTA_init(self) -> None:
        for index, neuron in enumerate(self.Layers):
            if index == 0:
                continue
            self.Matrix_DELTA[f"D{index}"] = np.zeros([self.Layers[index - 1], neuron])

    def Back_Propagation(self, Yi: np.ndarray or int):
        self.delta_init(Yi)
        for index in reversed(range(1, len(self.Layers) - 1)):
            cnt = self.Matrix_Output[f"O{index}"] * (1 - self.Matrix_Output[f"O{index}"])
            weight = self.Matrix_Weight[f"W{index + 1}"]
            delta = self.Matrix_delta[f"d{index + 1}"]
            self.Matrix_delta[f"d{index}"] = cnt * np.dot(weight, delta)


np.random.seed(7)
# [n, m] is the dataset of x and y in which
# n is the number of examples and m is the neurons
my_x = np.random.random([2, 2])
my_y = np.random.random([2, 1])
print(my_x)
print(my_y)

ai = NeuralNetwork(my_x, my_y)
ai.add_layer(3, 2)
ai.details()

ai.Weight_init()
pprint(ai.Matrix_Weight)

ai.Forward_Propagation(ai.X[0], NeuralNetwork.Sigmoid, NeuralNetwork.Sigmoid)
pprint(ai.Matrix_ProductSum)
pprint(ai.Matrix_Output)

ai.DELTA_init()
pprint(ai.Matrix_DELTA)

ai.Back_Propagation(ai.Y[0])
pprint(ai.Matrix_delta)

