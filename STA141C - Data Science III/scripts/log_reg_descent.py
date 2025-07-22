import numpy as np
import math
import time
from sklearn.model_selection import train_test_split
from .utils import load_data


def generateX(X):
    return np.c_[np.ones((len(X), 1)), X]

def get_initial_vec(X):
    return np.random.randn(X.shape[1] + 1, 1)

def sigmoid_function(X):
    return 1 / (1 + np.exp(-X))

def Logistic_Fit(X, y, learningrate, iterations):
    y_new = y.reshape(-1, 1)
    X_aug = generateX(X)
    theta = get_initial_vec(X)
    m = len(X)
    for _ in range(iterations):
        gradients = 2 / m * X_aug.T.dot(sigmoid_function(X_aug.dot(theta)) - y_new)
        theta = theta - learningrate * gradients
    return theta

def predict(theta, X):
    X_aug = generateX(X)
    logits = sigmoid_function(X_aug.dot(theta))
    return (logits >= 0.5).astype(int).ravel()

def accuracy_metric(y_true, y_pred):
    return (y_true == y_pred).mean()

def main():
    X, y, _ = load_data(standardize=False)
    accuracies = []
    start = time.time()
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )
        theta = Logistic_Fit(X_train, y_train, 0.1, 100)
        pred = predict(theta, X_test)
        accuracies.append(accuracy_metric(y_test, pred))
    end = time.time()
    print(f"Average accuracy: {np.mean(accuracies):.3f}")
    print(f"Training time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
