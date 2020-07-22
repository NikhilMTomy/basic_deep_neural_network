import numpy as np
import os

class Model:
    def initialize(self, train_x, train_y, test_x, test_y, learning_rate = 0.01, batch_size=5000, iterations = 200, debug=False):
        self.train_x = train_x
        self.train_y = train_y

        self.test_x = test_x
        self.test_y = test_y
        
        self.layer_count = 3
        self.layer_sizes = [
            self.train_x.shape[0],
            64,
            64,
            self.train_y.shape[0]
        ]
        self.W = [
            np.zeros((1,1)),
            np.random.randn(self.layer_sizes[0], self.layer_sizes[1]) * 0.01,
            np.random.randn(self.layer_sizes[1], self.layer_sizes[2]) * 0.01,
            np.random.randn(self.layer_sizes[2], self.layer_sizes[3]) * 0.01,
        ]
        self.B = [
            np.zeros((1,1)),
            np.zeros((self.layer_sizes[1], 1)),
            np.zeros((self.layer_sizes[2], 1)),
            np.zeros((self.layer_sizes[3], 1))
        ]
        self.alpha = learning_rate
        self.G = [
            None,
            self.relu,
            self.relu,
            self.sigmoid
        ]
        self.A = [None for i in range(self.layer_count+1)]
        self.Z = [None for i in range(self.layer_count+1)]

        self.iterations = iterations
        self.batch_size = batch_size
        self.debug = debug
        
    def print_info(self, val, var, force=False):
        if self.debug or force:
            print("{} : {}".format(var, val))
        
    def tanh(self, x, derivative=False):
        if derivative:
            return (1 - np.square(np.tanh(x)))
        else:
            return np.tanh(x)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        else:
            return 1 / (1+np.exp(-x))
    def relu(self, x, derivative=False):
        if derivative:
            return (x>0)
        else:
            return (x>0) * x
    def cost(self, derived_output, desired_output):
        return (-1/self.batch_size) * np.sum(desired_output * np.log(derived_output) + (1-desired_output) * np.log(1-derived_output)) 
    
    def propogate(self, X):
        self.A[0] = X
        self.Z[0] = X
        for i in range(1, self.layer_count+1):
            self.Z[i] = np.dot(self.W[i].T, self.A[i-1]) + self.B[i]
            self.A[i] = self.G[i](self.Z[i])
            self.print_info(self.A[i].shape, "A[{}].shape".format(i))
            self.print_info(self.W[i].shape, "W[{}].shape".format(i))
            self.print_info(self.B[i].shape, "B[{}].shape".format(i))
        return self.A[i]
    
    def optimize(self, Y):
        self.dZ = [None for x in range(self.layer_count+1)]
        self.dW = [None for x in range(self.layer_count+1)]
        self.dB = [None for x in range(self.layer_count+1)]
        for i in reversed(range(1, self.layer_count+1)):
            if i == self.layer_count: # output layer
                self.dZ[i] = self.A[i]-Y
            else:
                self.dZ[i] = np.dot(self.W[i+1], self.dZ[i+1]) * self.G[i](self.Z[i], derivative=True)
            self.dW[i] = (1/self.batch_size) * np.dot(self.A[i-1], self.dZ[i].T)
            self.dB[i] = (1/self.batch_size) * np.sum(self.dZ[i], axis=1, keepdims=True)
            self.W[i] -= self.dW[i] * self.alpha
            self.B[i] -= self.dB[i] * self.alpha
            self.print_info(self.dZ[i].shape, "dZ[{}].shape".format(i))
            self.print_info(self.dW[i].shape, "dW[{}].shape".format(i))
            self.print_info(self.dB[i].shape, "dB[{}].shape".format(i))
            self.print_info(self.W[i].shape, "W[{}].shape".format(i))
            self.print_info(self.B[i].shape, "B[{}].shape".format(i))
        
    def train(self):
        train_x_batches = np.split(self.train_x, self.train_x.shape[1]/self.batch_size, axis=1)
        train_y_batches = np.split(self.train_y, self.train_y.shape[1]/self.batch_size, axis=1)
        test_costs = []
        for i in range(self.iterations):
            for j in range(len(train_x_batches)):
                current_inputs = train_x_batches[j]
                desired_outputs = train_y_batches[j]
                derived_outputs = self.propogate(current_inputs)
                self.optimize(desired_outputs)
                cost = self.cost(derived_outputs, desired_outputs)
                print("Epoch : {0}/{1} :: Cost : {2}".format(i+1, self.iterations, cost), end="\r")
            print()
            accuracy, test_cost = self.test()
            test_costs.append(test_cost)
            if len(test_costs) > 5:
                if test_costs[-1] > test_costs[-2]:
                    print("test cost increased. overfitting?")
                    break
        return test_costs

    def test(self, test_x = None, test_y = None, tolerance=0.5):
        if test_x is None:
            test_x = self.test_x
        if test_y is None:
            test_y = self.test_y
        derived_output = self.propogate(test_x)
        accuracy = (np.abs(test_y - derived_output) < tolerance).all(axis=0).mean()
        self.print_info(accuracy, "Accuracy", True)
        test_cost = self.cost(derived_output, self.test_y)
        return accuracy, test_cost
    
    def save(self):
        if not os.path.exists("parameters"):
            os.mkdir("parameters")
        for i in range(len(self.W)):
            with open("parameters/B_{0}.bin".format(i), "wb") as f:
                np.save(f, self.B[i])
            with open("parameters/W_{0}.bin".format(i), "wb") as f:
                np.save(f, self.W[i])
            
    def load(self):
        if os.path.exists("parameters/W_0.bin"):
            for i in range(len(self.B)):
                with open("parameters/W_{0}.bin".format(i), "rb") as f:
                    self.W[i] = np.load(f)
                with open("parameters/B_{0}.bin".format(i), "rb") as f:
                    self.B[i] = np.load(f)
