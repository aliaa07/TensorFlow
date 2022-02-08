import numpy as np
import matplotlib.pyplot as plt

my_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
my_y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])


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
        self.lambda_ = _ if (_ := self.kwargs.get("lambda_")) is not None else 10 ** 3
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
        return np.round(np.dot(self.Matrix_Theta[f"Theta_l{layer - 1}"], self.Matrix_A[f"A_l{layer - 1}"]), 7)

    @staticmethod
    def logistic(z: np.ndarray):
        """
        :param z:
        :return: returns the g(z) of the z array
        """
        return np.round(1 / (1 + np.exp(-z)), 7)

    def A1_matrix_creator(self, Xi: np.array) -> None:
        """
        creates the first layer of input for NW
        it could be an array of floats
        :param Xi: the array to create first matrix with
        :return: None
        """
        input_neurons = self.layers[0][1][1]
        self.Matrix_A.update({"A_l1": np.ones([input_neurons + 1, 1])})
        for xi in range(1, input_neurons + 1):
            self.Matrix_A["A_l1"][xi] = Xi[xi - 1]

    def forward_propagation(self, Xi: np.array) -> np.array:
        """
        you have to input first a1 layer manually
        then running this means that for each layer an
        activation matrix has been created
        give it an X and returns the Y of it
        :param Xi: must be and np. array
        :return: it returns the result in form of an array
        """
        self.A1_matrix_creator(Xi)
        for layer in range(2, len(self.layers) + 1):
            activation = NeuralNetwork.logistic(self.Z(layer))
            self.Matrix_A.update(
                {f"A_l{layer}": activation})
            self.Matrix_A[f"A_l{layer}"][0] = 1
        return self.Matrix_A[f"A_l{len(self.layers)}"]

    def error(self, Yi: np.array) -> np.array:
        error = np.reshape(np.insert(Yi, 0, 1), [self.k + 1, 1]) - self.Matrix_A[f"A_l{len(self.layers)}"]
        self.Matrix_delta.update({f"d_l{len(self.layers)}": error})
        return error

    def JofTheta(self, Yi: np.array) -> float:
        """
        Calculates the j of theta for each example
        adding it to a list so the final result would be
        the sum of the list
        :param Yi: Yi of the example set
        :return: value of j for each set of examples
        """
        j = r = 0
        for k in range(1, self.k + 1):
            j += np.round((- ((Yi[k - 1] * np.log(self.Matrix_A[f"A_l{len(self.layers)}"][k][0])) +
                              ((1 - Yi[k - 1]) * np.log(1 - self.Matrix_A[f"A_l{len(self.layers)}"][k][0]))) / self.m), 7)
        for layer in self.Matrix_Theta:
            r += np.sum(self.Matrix_Theta[layer])
        r = np.round(((r ** 2) * self.lambda_) / (2 * self.m), 7)
        j += r
        self.J.append(j)
        return j

    def back_propagation(self) -> None:
        """
        conduct back prop
        :return: None
        """
        for layer in reversed(range(2, len(self.layers))):
            delta = np.round(np.dot(np.transpose(self.Matrix_Theta[f"Theta_l{layer}"]), self.Matrix_delta[f"d_l{layer + 1}"]), 7)
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

    def delta_cleaner(self) -> None:
        """
        clears the matrix of A
        :return: None
        """
        self.Matrix_delta.clear()

    def Run_one_time_through_examples(self) -> dict:
        self.DELTA_matrix_creator()
        for index, example in enumerate(self.X):
            prediction = self.forward_propagation(example)
            error = self.error(self.Y[index])
            J = self.JofTheta(self.Y[index])
            self.back_propagation()
            self.A_cleaner()
            self.delta_cleaner()
        return locals()

    def DELTA_cleaner(self) -> None:
        """
        clears the matrix of A
        :return: None
        """
        self.Matrix_DELTA.clear()

    def Gradiant(self) -> None:
        for layer in self.Matrix_DELTA:
            for j in range(self.Matrix_DELTA[layer].shape[1]):
                if j == 0:
                    self.Matrix_DELTA[layer] = self.Matrix_DELTA[layer] / self.m
                else:
                    self.Matrix_DELTA[layer] = (self.Matrix_DELTA[layer] / self.m) + (self.lambda_ * self.Matrix_Theta[f"Theta_l{layer[-1]}"])

    def Gradiant_decent(self, learning_rate) -> None:
        for layer in self.Matrix_Theta:
            self.Matrix_Theta[layer] -= learning_rate * self.Matrix_DELTA[f"DELTA_l{layer[-1]}"]

    def Train(self, learning_rate: float = 0.01, epochs=-1):
        self.Theta_matrix_creator()
        while epochs != 0:
            Details = self.Run_one_time_through_examples()
            self.Gradiant()
            self.Gradiant_decent(learning_rate)
            self.DELTA_cleaner()
            epochs -= 1
            print(f"____________________________________________________________________ Start of No.Epochs:     {epochs}"
                  f"__________________________________________________________________________\n",
                  Details, "\n", self.Matrix_Theta, "\n"
                  f"____________________________________________________________________ End of No.Epochs:     {epochs}"
                  f"____________________________________________________________________________\n")


ai = NeuralNetwork(my_x, my_y, lambda_=100)
ai.add_layer(2)
ai.Train(learning_rate=0.1, epochs=200)
print(ai.forward_propagation(np.array([3])))
