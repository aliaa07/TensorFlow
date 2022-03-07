from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)


class NeuralNetwork:
    J = list()
    layers = list()
    Matrix_Theta = dict()
    Matrix_delta = dict()
    Matrix_DELTA = dict()
    Matrix_Activation = dict()
    m = k = features = 0

    def __init__(self, X=None, Y=None, dataset=None, epsilon=(10 ** -4)):
        self.X = X
        self.Y = Y
        self.dataset = dataset
        if self.X.ndim == self.Y.ndim and self.X.shape[0] == self.Y.shape[0]:
            if self.X.ndim == 1:
                print("*** Note: we have increased the dims of X by one in axis=-1 ***\n")
                self.X = np.expand_dims(self.X, axis=-1)
                self.Y = np.expand_dims(self.Y, axis=-1)
        else:
            raise ValueError("The input X and Y are not of the same dimension\n"
                             "or they are not of the same size each Yi must satisfy on Xi")
        self.epsilon = epsilon
        self.m = self.X.shape[0]
        self.features = self.X.shape[1]
        self.k = self.Y.shape[1]
        self.layers.append(self.X.shape[1])  # input neurons (initializers)
        self.layers.append(self.k)  # output neurons (classes)

    def add_layer(self, *args, list_of_layers=None):
        if list_of_layers is not None:
            args = list_of_layers
        for layer in args:
            self.layers.insert(-1, layer)

    def Theta_init(self):
        for index in range(1, len(self.layers)):
            self.Matrix_Theta[f"T{index}"] = np.random.random([self.layers[index], self.layers[index - 1] + 1]) * (self.epsilon * 2) - self.epsilon

    def Activation_init(self, Xi):
        self.Matrix_Activation.clear()
        for index in range(1, len(self.layers) + 1):
            self.Matrix_Activation[f"A{index}"] = np.zeros([self.layers[index - 1], 1])
            if index == 1:
                self.Matrix_Activation[f"A{index}"] = np.expand_dims(Xi, axis=-1)

    def error_screener(self, error, loss=None):
        pass

    def delta_init(self, Yi, loss=None):
        self.Matrix_delta.clear()
        error = 0
        for index in range(2, len(self.layers) + 1):
            if index != len(self.layers):
                self.Matrix_delta[f"d{index}"] = np.zeros([self.layers[index - 1] + 1, 1])
            else:
                error = self.Matrix_delta[f"d{index}"] = self.Matrix_Activation[f"A{len(self.layers)}"] - np.expand_dims(Yi, axis=-1)
                self.error_screener(error, loss)
        return error

    def DELTA_init(self):
        self.Matrix_DELTA.clear()
        for index in range(1, len(self.layers)):
            self.Matrix_DELTA[f"D{index}"] = np.zeros([self.layers[index], self.layers[index - 1] + 1])

    @staticmethod
    def adding_bias(Matrix, trans=False):
        if not trans:
            Activation_with_bias = np.transpose(np.expand_dims(np.insert(Matrix, 0, 1), axis=0))
        else:
            Activation_with_bias = np.expand_dims(np.insert(Matrix, 0, 1), axis=0)
        return Activation_with_bias

    @staticmethod
    def logistic(z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, Xi):
        self.Activation_init(Xi)
        for index in range(2, len(self.layers) + 1):
            self.Matrix_Activation[f"A{index}"] = self.logistic(np.dot(self.Matrix_Theta[f"T{index - 1}"],
                                                                       self.adding_bias(self.Matrix_Activation[f"A{index - 1}"])))
        return self.Matrix_Activation[f"A{len(self.layers)}"]

    def predict(self, Xi, whole_activation_matrix=False, ones_and_zeros=(False, 0.5)):
        theta = deepcopy(self.Matrix_Theta)
        self.Activation_init(Xi)
        activation = deepcopy(self.Matrix_Activation)
        for layer in activation:
            activation[layer] *= 0
            if int(layer[-1]) == 1:
                activation[layer] = Xi
        for index in range(2, len(self.layers) + 1):
            activation[f"A{index}"] = np.transpose(self.logistic(np.dot(theta[f"T{index - 1}"],
                                                                        self.adding_bias(activation[f"A{index - 1}"]))))
        if not ones_and_zeros[0]:
            if whole_activation_matrix:
                return activation
            else:
                return activation[f"A{len(self.layers)}"]
        else:
            if activation[f"A{len(self.layers)}"] > 0.5:
                return 1
            else:
                return 0

    def back_propagation(self, Yi, loss=None):
        error = self.delta_init(Yi, loss)
        for layer in reversed(range(2, len(self.layers))):
            refined_a = self.adding_bias(self.Matrix_Activation[f"A{layer}"])
            gradiant_a = refined_a * (1 - refined_a)
            if (layer + 1) == len(self.layers):
                delta = np.dot(np.transpose(self.Matrix_Theta[f"T{layer}"]), self.Matrix_delta[f"d{layer + 1}"])
            else:
                delta = np.dot(np.transpose(self.Matrix_Theta[f"T{layer}"]), self.Matrix_delta[f"d{layer + 1}"][1:])
            self.Matrix_delta[f"d{layer}"] = delta * gradiant_a
        for layer in range(1, len(self.layers)):
            refined_a = self.adding_bias(self.Matrix_Activation[f"A{layer}"])
            if layer != len(self.layers) - 1:
                self.Matrix_DELTA[f"D{layer}"] += np.dot(self.Matrix_delta[f"d{layer + 1}"][1:], np.transpose(refined_a))
            else:
                self.Matrix_DELTA[f"D{layer}"] += np.dot(self.Matrix_delta[f"d{layer + 1}"], np.transpose(refined_a))
        return error

    def learn_examples(self, optimizer=None, loss=None):
        self.DELTA_init()
        learning_result = list()
        for example in range(self.m):
            Hypothesis = self.forward_propagation(self.X[example])
            error = self.back_propagation(self.Y[example])
            self.Matrix_Activation.clear()
            self.Matrix_delta.clear()
            learning_result.append((Hypothesis, error, self.Y[example]))
        return learning_result

    def gradient(self):
        for layer in range(1, len(self.layers)):
            self.Matrix_DELTA[f"D{layer}"] /= 1 / self.m

    def gradient_decent(self, learning_rate=0.01):
        for layer in range(1, len(self.layers)):
            self.Matrix_Theta[f"T{layer}"] -= learning_rate * self.Matrix_DELTA[f"D{layer}"]

    def JofTheta(self, learning_result_tuple, regularization=(False, 0)):
        J = 0
        for detail in learning_result_tuple:
            H, Yi = detail[0], detail[2]
            for Hk, Yik in zip(H, Yi):
                J += (Yik * np.log(Hk)) + ((1 - Yik) * np.log(1 - Hk))
        J /= - (1 / self.m)
        if regularization[0]:
            r = 0
            for layer in self.Matrix_Theta:
                r += (regularization[1] / (2 * self.m)) * np.sum(self.Matrix_Theta[layer][:, 1:] ** 2)
            J += r
        return J

    def Train(self, learning_rate=0.01, epochs=0, detailed=True, optimizer=None, loss=None, regularization=(False, 0)):
        self.Theta_init()
        counter = 1
        while counter <= epochs:
            Details = self.learn_examples(optimizer, loss)
            self.J.append(self.JofTheta(Details, regularization))
            self.gradient()
            self.gradient_decent(learning_rate)
            if detailed:
                # doesn't need ,
                print(f"____________________________________________________________________ Start of No.Epochs:     {counter}"
                      f" __________________________________________________________________________\n",
                      Details, "\n", self.Matrix_Theta, "\n",
                      f"******************************************************************** End of No.Epochs:     {counter}"
                      f" ****************************************************************************\n")
            self.Matrix_DELTA.clear()
            counter += 1

    def plot_J(self) -> None:
        x_number_values = [index for index, item in enumerate(self.J)]
        y_number_values = self.J
        plt.xlabel("No.iteration", fontsize=10)
        plt.ylabel("J(theta)", fontsize=10)
        plt.plot(x_number_values, y_number_values)
        plt.show()


# my_x = np.round(np.random.random([3, 1]), 2)
# my_y = np.round(np.random.random([3, 2]), 2)
n = 20
my_x = np.array([i for i in range(n)])
my_y = np.array([1 if i % 2 == 0 else 0 for i in range(n)])

ai = NeuralNetwork(my_x, my_y, epsilon=0.001)
ai.add_layer(list_of_layers=[10 for i in range(20)])
ai.Train(epochs=10, learning_rate=0.01, detailed=False)

print(ai.predict(4))
print(ai.predict(5))
print(ai.predict(6))
print(ai.predict(7))
ai.plot_J()
