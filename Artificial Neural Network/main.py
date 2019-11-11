import numpy as np

from neural_network import NeuralNetwork

# input
x = np.array([
    [30, 40, 50],
    [40, 50, 20],
    [50, 20, 15],
    [20, 15, 60],
    [15, 60, 70],
    [60, 70, 50]
], dtype=np.float64)

# Expected output
y = np.array([20, 15, 60, 70, 50, 40], dtype=np.float64)

size_of_learn_sample = int(len(x)*0.9)
print('Size of Learning Sample: ', size_of_learn_sample, '\n')

nn = NeuralNetwork(x, y, 0.5)

# NN.print_matrices()
nn.train()
nn.print_matrices()

