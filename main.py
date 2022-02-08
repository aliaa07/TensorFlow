import numpy as np
import matplotlib.pyplot as plt

# my_x = np.array([5, 10, 15, 20, 25])
# my_y = np.array([10, 20, 30, 40, 50])
my_x = np.random.random([200, 6])
my_y = np.random.random([200, 6])


class NeuralNetwork:
    Matrix_Theta = dict()
    Matrix_delta = dict()
    Matrix_DELTA = dict()
    Matrix_A = dict()
    layers = list()
    J = list()
    D = dict
    m = 0
    k = 0

    def __init__(self, X: np.ndarray = None, Y: np.ndarray = None, **kwargs) -> None:
        self.Y = Y
        self.X = X
        if self.X.ndim == self.Y.ndim:
            if self.X.ndim == 1:
                print("*** Note: we have increased the dims of X by one in axis=-1 ***\n")
                self.X = np.expand_dims(self.X, axis=-1)
                self.Y = np.expand_dims(self.Y, axis=-1)
        else:
            raise TypeError("The input X and Y are of the same dim")
        self.kwargs = kwargs
        self.epsilon = _ if (_ := self.kwargs.get("epsilon")) is not None else 10 ** -5
        self.lambda_ = _ if (_ := self.kwargs.get("lambda_")) is not None else 100
        self.layers.insert(0, (self.X.ndim, self.X.shape))
        self.layers.append((self.Y.ndim, self.Y.shape))
        self.m = self.X.shape[0]
        self.k = self.Y.shape[1]

    def add_layer(self, neurons: int = 5) -> None:
        """
        initialize a layer to the NW with n number of neurons
        :param neurons: neurons of the initialized layer
        :return: None
        """
        self.layers.insert(-1, (1, (1, neurons)))

    def Theta_matrix_creator(self) -> None:
        """
        creates the theta matrix
        ** must be called just and only ONCE **
        :return: None
        """
        for index, layer in enumerate(self.layers):
            if index == 0:
                continue
            name = f"Theta_l{index}"
            C = layer[1][1] + 1
            P = self.layers[index - 1][1][1] + 1
            self.Matrix_Theta.update({
                name: (np.random.random([C, P]) * (self.epsilon * 2) - self.epsilon)
            })

    def DELTA_matrix_creator(self) -> None:
        for index, layer in enumerate(self.layers):
            if index == 0:
                continue
            name = f"DELTA_l{index}"
            C = layer[1][1] + 1
            P = self.layers[index - 1][1][1] + 1
            self.Matrix_DELTA.update({name: np.zeros([C, P])})

    def Z(self, layer: int) -> np.array:
        """
        gets a layer from z2 to final layer and returns the matrix
        of the calculated layer
        :param layer: layer starting form 2 to the L
        :return: matrix of calculated layer
        """
        return np.dot(self.Matrix_Theta[f"Theta_l{layer - 1}"], self.Matrix_A[f"A_l{layer - 1}"])

    @staticmethod
    def logistic(z: np.float64):
        """
        :param z:
        :return: returns the g(z) of the z array
        """
        return np.round(1 / (1 + np.exp(-z)), 2)

    def A1_matrix_creator(self, Xi: int) -> None:
        """
        creates the first layer of input for NW
        it could be an array of floats
        :param Xi: the array to create first matrix with
        :return: None
        """
        input_neurons = self.layers[0][1][1]
        self.Matrix_A.update({"A_l1": np.ones([input_neurons + 1, 1])})
        for xi in range(1, input_neurons + 1):
            self.Matrix_A["A_l1"][xi] = self.X[Xi][xi - 1]

    def forward_propagation(self, Xi: int) -> None:
        """
        you have to input first a1 layer manually
        then running this means that for each layer an
        activation matrix has been created
        :return:
        """
        self.A1_matrix_creator(Xi)
        for layer in range(2, len(self.layers) + 1):
            activation = NeuralNetwork.logistic(self.Z(layer))
            self.Matrix_A.update(
                {f"A_l{layer}": activation})
            self.Matrix_A[f"A_l{layer}"][0] = 1

    def error(self, Xi: int):
        error = np.reshape(np.insert(self.Y[Xi], 0, 1), [self.k + 1, 1]) - self.Matrix_A[f"A_l{len(self.layers)}"]
        self.Matrix_delta.update({f"d_l{len(self.layers)}": error})

    def JofTheta(self, Xi: int) -> float:
        """
        Calculates the j of theta for each example
        adding it to a list so the final result would be
        the sum of the list
        :param Xi: x i of the example set
        :return: value of j for each set of examples
        """
        j = r = 0
        Yi = self.Y[Xi]
        for k in range(1, self.k + 1):
            j += (- ((Yi[k - 1] * np.log(self.Matrix_A[f"A_l{len(self.layers)}"][k][0])) +
                     ((1 - Yi[k - 1]) * np.log(1 - self.Matrix_A[f"A_l{len(self.layers)}"][k][0]))) / self.m)
        for layer in self.Matrix_Theta:
            r += np.sum(self.Matrix_Theta[layer])
        r = ((r ** 2) * self.lambda_) / (2 * self.m)
        j += r
        self.J.append(j)
        return j

    def back_propagation(self, Xi: int):
        for layer in reversed(range(2, len(self.layers))):
            delta = np.dot(np.transpose(self.Matrix_Theta[f"Theta_l{layer}"]), self.Matrix_delta[f"d_l{layer + 1}"])
            a = self.Matrix_A[f"A_l{layer}"] * (1 - self.Matrix_A[f"A_l{layer}"])
            self.Matrix_delta.update({f"d_l{layer}": delta * a})
        for layer in self.Matrix_DELTA:
            self.Matrix_DELTA[layer] += np.dot(self.Matrix_delta[f"d_l{str(int(layer[-1]) + 1)}"], np.transpose(self.Matrix_A[f"A_l{layer[-1]}"]))

    def A_cleaner(self) -> None:
        """
        clears the matrix of A
        :return: None
        """
        self.Matrix_A.clear()

    def Run(self) -> dict[str, np.ndarray]:
        self.DELTA_matrix_creator()
        for example in range(self.X.shape[0]):
            self.back_propagation(example)
            self.JofTheta(example)
            self.A_cleaner()
        return locals()

    def Gradiant(self) -> None:
        details = self.Run()
        for layer in self.Matrix_DELTA:
            for j in range(self.Matrix_DELTA[layer].shape[1]):
                if j == 0:
                    self.Matrix_DELTA[layer] = ((1 / self.m) * self.Matrix_DELTA[layer])
                else:
                    self.Matrix_DELTA[layer] = ((1 / self.m) * self.Matrix_DELTA[layer]) + (self.lambda_ * self.Matrix_Theta[f"Theta_l{layer[-1]}"])
        return details

    def Gradiant_decent(self, learning_rate: float = 0.01, epochs=-1):
        self.Theta_matrix_creator()
        while epochs != 0:
            details = self.Gradiant()
            for layer in self.Matrix_Theta:
                self.Matrix_Theta[layer] -= learning_rate * self.Matrix_DELTA[f"DELTA_l{layer[-1]}"]
            epochs -= 1
            print(details)


ai = NeuralNetwork(my_x, my_y)
ai.add_layer(3)
ai.Gradiant_decent(0.01, 100)

