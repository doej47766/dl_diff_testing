import requests, gzip, os, hashlib
import numpy as np

# https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb
def fetchMNISTFromURL(url):
    fp = os.path.join('./datasets/mnist', hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
            print(type(gzip.decompress(data)))
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def fetchMNIST():
    X_train_mnist = fetchMNISTFromURL("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_train_mnist = fetchMNISTFromURL("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test_mnist = fetchMNISTFromURL("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_test_mnist = fetchMNISTFromURL("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    
    return X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist