import pandas as pd

from preprocess import load_data_set, get_encoder, filter_string
from train import forward
import numpy as np

def load_weights():
    W = pd.read_csv("results/best_weights.csv", header=None)
    W = W.as_matrix()
    return W


def load_bias():
    b = pd.read_csv("results/bias.csv", header=None)
    b = b.as_matrix()
    return b


def encoder():
    print(load_data_set()[0].shape)
    return get_encoder(load_data_set()[0])


np.set_printoptions(threshold=np.nan)


def get_encoded_data(string, encoder):
    string = filter_string(string)
    data = encoder.transform([string]).toarray()
    return data


def get_sentiment(word):
    x = get_encoded_data(word, enc)
    sentiment = forward(x, W, bias)
    print(sentiment)
    return round(sentiment[0][0])


W = load_weights()
bias = load_bias()
enc = encoder()

while True:
    print("Paste comment")
    string = input()
    result = get_sentiment(string)
    print("Negative" if result == 0 else "Positive")
