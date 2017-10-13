import re
import os
import numpy as np
from bs4 import BeautifulSoup as Soup
from sklearn.feature_extraction.text import CountVectorizer as encoder
from sklearn.utils import shuffle

np.set_printoptions(threshold=np.nan)


def load_data(path):
    print("load_data: " + path)
    file = open(path)
    soup = Soup(file, 'lxml')
    text_set = soup.find_all("review_text")
    words_list = []
    data = 0
    for elem in text_set:
        element = filter_string(str(elem))
        words_list.append(element)
        data = np.array(words_list)

    return data


def filter_string(element):
    element = re.sub(r'([^\s\w]|_)+', '', element)
    element = re.sub(r'[0-9]', '', element)
    element = re.sub(r'\n', '', element)
    element = re.sub(r'\b\w{1,4}\b', '', element)
    element = re.sub(' +', ' ', element)
    element = element.replace('reviewtext', '')
    element = element.lower()
    return element


def get_encoder(dataset):
    print("get_encoder")
    enc = encoder()
    enc.fit(dataset)
    return enc


# TODO in preprocessing of test set (or sample) get list of words and change words to its closest in dictionary
def load_data_set():
    print("load_data_set")
    X_N_ELECTRONICS = load_data("Data/electronics/negative.review")
    # X_N_DVD = load_data("Data/dvd/negative.review")
    X_N_BOOKS = load_data("Data/books/negative.review")
    # X_N_KITCHEN = load_data("Data/kitchen_housewares/negative.review")

    X_N = np.hstack((X_N_BOOKS, X_N_ELECTRONICS))
    # X_N = np.hstack((X_N, X_N_DVD))
    # X_N = np.hstack((X_N, X_N_KITCHEN))

    X_P_ELECTRONICS = load_data("Data/electronics/positive.review")
    # X_P_DVD = load_data("Data/dvd/positive.review")
    X_P_BOOKS = load_data("Data/books/positive.review")
    # X_P_KITCHEN = load_data("Data/kitchen_housewares/positive.review")

    X_P = np.hstack((X_P_BOOKS, X_P_ELECTRONICS))
    # X_P = np.hstack((X_P, X_P_DVD))
    # X_P = np.hstack((X_P, X_P_KITCHEN))

    X = np.hstack((X_N, X_P))
    N = X.shape[0]
    Y = np.array([0] * int(N / 2) + [1] * int(N / 2))
    X, Y = shuffle(X, Y)

    return X, Y


def encode_string(line, encoder):
    arr = encoder.transform([line]).toarray()
    return arr


from tools import print_progress
from tools import serialize

def generate_encoded_data_file():
    print("generate_encoded_data_file")
    X, Y = load_data_set()
    Y = np.reshape(Y, (Y.shape[0], 1))
    encoder = get_encoder(X)
    serialize(encoder(), "encoder")
    X_encoded = encode_string("", encoder)

    for i in range(0, X.shape[0], 1):
        print_progress(i, X.shape[0], prefix='Progress:', suffix='Complete', bar_length=50)
        line = encode_string(X[i], encoder)
        X_encoded = np.vstack((X_encoded, line))

    X_encoded = X_encoded[1:, :]
    X_encoded = np.hstack((X_encoded, Y))
    if not os.path.exists("preprocessed"):
        os.makedirs("preprocessed")
    np.savetxt("preprocessed/data.csv", X_encoded, delimiter=',', fmt="%d")


def read_encoded_csv():
    print("read_encoded_csv")
    import pandas as pd
    data = pd.read_csv("preprocessed/data.csv", header=None)
    data = data.as_matrix()
    X = data[:, :-1]
    Y = data[:, -1]
    Y = np.reshape(Y, (Y.shape[0], 1))
    return X, Y


