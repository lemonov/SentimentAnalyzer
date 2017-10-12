from preprocess import read_encoded_csv as get_data
import numpy as np
import matplotlib.pyplot as plt
import os


def initialize():
    X, T = get_data()
    T = T.astype(float)
    W = np.random.randn(X.shape[1], 1)
    b = 0
    return X, T, W, b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, W, b):
    return sigmoid(X.dot(W) + b)


def classification_rate(T, Y):
    print(Y.shape)
    print(T.shape)

    return (Y == T).mean()


def cross_entropy(T, Y):
    E = 0
    for i in range(0, T.shape[0], 1):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


def train():
    X, T, W, b = initialize()

    Y = forward(X, W, b)
    lambda_l1 = 0
    lambda_l2 = 0
    learning_rate = 0.001
    bias_list = []
    CEE = []

    for i in range(0, 1000, 1):
        print("Epoch: " + str(i))
        Y = forward(X, W, b)
        delta = X.T.dot(T - Y) - lambda_l2 * W - lambda_l1 * np.sign(W) # elastic net
        b = b + learning_rate * (T - Y).sum()
        bias_list.append(b)
        W = W + learning_rate * delta
        CEE.append(cross_entropy(T, Y).mean())

    plt.plot(bias_list)
    plt.show()

    plt.plot(CEE)
    plt.show()

    print(W.shape)
    print(X.shape)


    if not os.path.exists("results"):
        os.makedirs("results")
    np.savetxt("results/best_weights.csv", W, delimiter=',')

    if not os.path.exists("results"):
        os.makedirs("results")
    file = open("results/bias.csv", "w")
    file.write(str(b))
    file.close()

    print(classification_rate(np.round(T), Y))


# train()