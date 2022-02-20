import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class NeuralNetwork:
    layers = list()
    Matrix_A = dict()
    Matrix_Theta = dict()
    Matrix_delta = dict()
    Matrix_DELTA = dict()
    j_of_on_round_through_exps = list()
    Matrix_Theta_for_gradient_checking = dict()
    Matrix_DELTA_for_gradient_checking = dict()
    Matrix_A_for_gradient_checking = dict()
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
        self.lambda_ = _ if (_ := self.kwargs.get("lambda_")) is not None else 2
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

    def DELTA_matrix_creator(self, Matrix_name: dict) -> None:
        for index, layer in enumerate(self.layers):
            if index == 0:
                continue
            name = f"DELTA_l{index}"
            C = layer[1][1] + 1
            P = self.layers[index - 1][1][1] + 1
            Matrix_name.update({name: np.zeros([C, P])})

    def Z(self, layer: int, matrix_t: np.array, matrix_a: np.array) -> np.array:
        """
        gets a layer from z2 to final layer and returns the matrix
        of the calculated layer
        :param matrix_a: Matrix of activation
        :param matrix_t: Matrix of theta
        :param layer: layer starting form 2 to the L
        :return: matrix of calculated layer
        """
        return np.dot(matrix_t[f"Theta_l{layer - 1}"], matrix_a[f"A_l{layer - 1}"])

    @staticmethod
    def logistic(z: np.ndarray):
        """
        :param z:
        :return: returns the g(z) of the z array
        """
        return 1 / (1 + np.exp(-z))

    def A1_matrix_creator(self, Xi: np.array, matrix_a: np.array) -> None:
        """
        creates the first layer of input for NW
        it could be an array of floats
        :param matrix_a: the activation matrix
        :param Xi: the array to create first matrix with
        :return: None
        """
        input_neurons = self.layers[0][1][1]
        matrix_a.update({"A_l1": np.ones([input_neurons + 1, 1])})
        for xi in range(1, input_neurons + 1):
            matrix_a["A_l1"][xi] = Xi[xi - 1]

    def forward_propagation(self, Xi: np.array, **kwargs) -> np.array:
        """
        you have to input first a1 layer manually
        then running this means that for each layer an
        activation matrix has been created
        give it an X and returns the Y of it
        :param Xi: must be and np. array
        :return: it returns the result in form of an array
        """
        if kwargs.get("epsilon") is not None:
            self.Matrix_A_for_gradient_checking.clear()
            self.Matrix_Theta_for_gradient_checking.clear()
            self.Matrix_Theta_for_gradient_checking = deepcopy(self.Matrix_Theta)
            self.Matrix_Theta_for_gradient_checking[f"Theta_l{kwargs.get('l')}"][kwargs.get('i'), kwargs.get('j')] += kwargs.get("epsilon")
            self.A1_matrix_creator(Xi, self.Matrix_A_for_gradient_checking)
            for layer in range(2, len(self.layers) + 1):
                activation = NeuralNetwork.logistic(self.Z(layer, self.Matrix_Theta_for_gradient_checking, self.Matrix_A_for_gradient_checking))
                self.Matrix_A_for_gradient_checking.update({f"A_l{layer}": activation})
                self.Matrix_A_for_gradient_checking[f"A_l{layer}"][0] = 1
            return self.Matrix_A_for_gradient_checking[f"A_l{len(self.layers)}"]
        else:
            self.A1_matrix_creator(Xi, self.Matrix_A)
            for layer in range(2, len(self.layers) + 1):
                activation = NeuralNetwork.logistic(self.Z(layer, self.Matrix_Theta, self.Matrix_A))
                self.Matrix_A.update(
                    {f"A_l{layer}": activation})
                self.Matrix_A[f"A_l{layer}"][0] = 1
            return self.Matrix_A[f"A_l{len(self.layers)}"]

    def error(self, Yi: np.array) -> np.array:
        error = np.reshape(np.insert(Yi, 0, 1), [self.k + 1, 1]) - self.Matrix_A[f"A_l{len(self.layers)}"]
        self.Matrix_delta.update({f"d_l{len(self.layers)}": error})
        return error

    def J_of_theta_for_one_examples(self, Yi: np.array) -> None:
        """
        Calculates the j of theta for each example
        adding it to a list so the final result would be
        the sum of the list
        :param Yi: Yi of the example set
        :return: value of j for each set of examples
        """
        j = r = 0
        for k in range(1, self.k + 1):
            j += (- ((Yi[k - 1] * np.log(self.Matrix_A[f"A_l{len(self.layers)}"][k][0])) +
                     ((1 - Yi[k - 1]) * np.log(1 - self.Matrix_A[f"A_l{len(self.layers)}"][k][0]))) / self.m)
        for layer in self.Matrix_Theta:
            r += np.sum(self.Matrix_Theta[layer][1:, :])
        r = ((r ** 2) * self.lambda_) / (2 * self.m)
        j += r
        self.j_of_on_round_through_exps.append(j)

    def back_propagation(self) -> None:
        """
        conduct back prop
        :return: None
        """
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

    def delta_cleaner(self) -> None:
        """
        clears the matrix of A
        :return: None
        """
        self.Matrix_delta.clear()

    def Run_one_time_through_examples(self) -> dict:
        self.DELTA_matrix_creator(self.Matrix_DELTA)
        for index, example in enumerate(self.X):
            prediction = self.forward_propagation(example)
            error = self.error(self.Y[index])
            self.J_of_theta_for_one_examples(self.Y[index])
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
                    self.Matrix_DELTA[layer][1:, j] /= self.m
                else:
                    self.Matrix_DELTA[layer][1:, j] = \
                        (self.Matrix_DELTA[layer][1:, j] +
                         (self.lambda_ * self.Matrix_Theta[f"Theta_l{layer[-1]}"][1:, j])) / self.m

    def Gradiant_checking(self, epsilon) -> dict:
        self.DELTA_matrix_creator(self.Matrix_DELTA_for_gradient_checking)
        for layer in self.Matrix_Theta:
            for i in range(self.Matrix_Theta[layer].shape[0]):
                for j in range(self.Matrix_Theta[layer].shape[1]):
                    J_plus = 0
                    J_minus = 0
                    for xi, yi in zip(self.X, self.Y):
                        h_plus = self.forward_propagation(xi, epsilon=epsilon, l=layer[-1], i=i, j=j)
                        h_minus = self.forward_propagation(xi, epsilon=-epsilon, l=layer[-1], i=i, j=j)
                        for k in range(self.k):
                            J_plus += - ((yi[k] * np.log(h_plus[k + 1])) + ((1 - yi[k]) * np.log(1 - h_plus[k + 1]))) / self.m
                        for k in range(self.k):
                            J_minus += - ((yi[k] * np.log(h_minus[k + 1])) + ((1 - yi[k]) * np.log(1 - h_minus[k + 1]))) / self.m
                    self.Matrix_DELTA_for_gradient_checking[f"DELTA_l{layer[-1]}"][i][j] = (J_plus - J_minus) / 2 * epsilon
        error_of_gradiant = dict()
        for layer in self.Matrix_DELTA_for_gradient_checking:
            error = abs(self.Matrix_DELTA[layer] - self.Matrix_DELTA_for_gradient_checking[layer])
            error_of_gradiant.update({layer: error})
        return error_of_gradiant

    def Gradiant_decent(self, learning_rate, use_gradiant_of_GC=False) -> None:
        if not use_gradiant_of_GC:
            for layer in self.Matrix_Theta:
                self.Matrix_Theta[layer] -= learning_rate * self.Matrix_DELTA[f"DELTA_l{layer[-1]}"]
        else:
            for layer in self.Matrix_Theta:
                self.Matrix_Theta[layer] -= learning_rate * self.Matrix_DELTA_for_gradient_checking[f"DELTA_l{layer[-1]}"]

    def J_of_theta(self) -> np.array:
        j = np.sum(self.j_of_on_round_through_exps)
        self.J.append(j)
        self.j_of_on_round_through_exps.clear()
        return j

    def Train(self, learning_rate: float = 0.01, epochs=-1, gradiant_checking=False, epsilon=(10 ** -4), use_gradiant_of_GC=False):
        self.Theta_matrix_creator()
        while epochs != 0:
            Details = self.Run_one_time_through_examples()
            self.J_of_theta()
            self.Gradiant()
            if gradiant_checking:
                self.Gradiant_checking(epsilon)
            self.Gradiant_decent(learning_rate, use_gradiant_of_GC)
            self.Matrix_DELTA_for_gradient_checking.clear()
            self.DELTA_cleaner()
            epochs -= 1
            # doesn't need ,
            print(f"____________________________________________________________________ Start of No.Epochs:     {epochs}"
                  f" __________________________________________________________________________\n",
                  Details, "\n", self.Matrix_Theta, "\n",
                  f"____________________________________________________________________ End of No.Epochs:     {epochs}"
                  f" ____________________________________________________________________________\n")

    def plot_J(self) -> None:
        x_number_values = [index for index, item in enumerate(self.J)]
        y_number_values = self.J
        plt.xlabel("No.iteration", fontsize=10)
        plt.ylabel("J(theta)", fontsize=10)
        plt.plot(x_number_values, y_number_values)
        plt.show()


# my_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# my_y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])

my_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
my_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

ai = NeuralNetwork(my_x, my_y)
ai.add_layer(3)
ai.Train(epochs=1000)
print(ai.forward_propagation(np.array([12])))
ai.plot_J()

