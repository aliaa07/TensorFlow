import numpy as np
import matplotlib.pyplot as plt


def control(num):
    num = str(num)
    n = len(num)
    return 10 ** -(n - 1)


def Sigma(x, y, phi: float) -> float:
    if len(x) == len(y):
        m = len(x)
        add_up = np.sum(((phi * x) - y) * x)
        return add_up / m
    else:
        return 0


def J_of_phi(x, y, phi: float) -> float:
    if len(x) == len(y):
        m = len(x)
        add_up = np.sum(((phi * x) - y) * x)
        return add_up / (m * 2)
    else:
        return 0


def Gradiant_decent(x, y, phi, learning_rate):
    mae = Sigma(x, y, phi)
    j_of_phi = J_of_phi(x, y, phi)
    refined_phi = phi - (learning_rate * mae)
    return locals()


class LinerReg:
    ML_dict = dict()
    slope = 0
    counter = list()
    GD = list()

    def __init__(self, len_of_list, X=np.array([]), Y=np.array([]), start_off=0, epochs=0, Initial_phi=0, Learning_rate=(10 * (10 ** -3))):
        self.len_of_list = len_of_list
        self.start_off = start_off
        self.X = np.array(X)
        self.Y = np.array(Y)
        if self.X.size == 0 or self.Y.size == 0 or self.X.size != self.Y.size:
            self.X = self.rand_num_creator(self.len_of_list, self.start_off)
            self.Y = self.rand_num_creator(self.len_of_list, self.start_off)
        self.epochs = epochs
        self.Initial_phi = Initial_phi
        self.Learning_rate = Learning_rate

        accurate_to_lr = False
        refining_phi = self.Initial_phi
        counter = 0
        while not accurate_to_lr:
            LinerReg.counter.append(counter)
            counter += 1
            GD = Gradiant_decent(self.X, self.Y, refining_phi, self.Learning_rate)
            refining_phi = GD["refined_phi"]
            LinerReg.GD.append(GD["j_of_phi"])
            if abs(GD["mae"]) <= self.Learning_rate / (10 ** 10) or counter == self.epochs:
                LinerReg.ML_dict = GD
                del LinerReg.ML_dict["x"]
                del LinerReg.ML_dict["y"]
                LinerReg.slope = refining_phi
                accurate_to_lr = True

    def F(self, x) -> float:
        return LinerReg.slope * x

    def plot(self):
        plt.scatter(self.X, self.Y)
        plt.plot([0, self.X[-1]], [self.F(0), self.F(self.X[-1])], color="red")
        plt.show()

    @staticmethod
    def plot_J_of_phi():
        plt.scatter(LinerReg.counter, LinerReg.GD)
        plt.show()

    @staticmethod
    def rand_num_creator(len_of_list, start_off):
        random_list = list()
        for i in range(len_of_list):
            var = np.random.randint(start_off, len_of_list * 10)
            while var in random_list:
                var = np.random.randint(start_off, len_of_list * 10)
            random_list.append(var)
        return np.array(sorted(random_list))


AI1 = LinerReg(len_of_list=10, epochs=50, Learning_rate=0.0001)
print(AI1.slope)
print(AI1.ML_dict)
AI1.plot()
# AI1.plot_J_of_phi()
