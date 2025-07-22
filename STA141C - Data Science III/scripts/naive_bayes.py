import numpy as np
import time
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import train_test_split
from .utils import load_data


def max_class(x, prior_list, mean_list, variance_list):
    likelihoods = []
    for idx in range(2):
        num = np.exp((-1/2) * ((x - mean_list[idx]) ** 2) / (2 * variance_list[idx]))
        den = np.sqrt(2 * np.pi * variance_list[idx])
        likelihoods.append(num / den)
    post = [np.log(prior_list[0]) + np.sum(np.log(likelihoods[0])),
            np.log(prior_list[1]) + np.sum(np.log(likelihoods[1]))]
    return np.argmax(post)


def Parallel_NB(X_train, X_test, y_train):
    n = len(X_train)
    m = X_train.shape[1]
    prior_list = np.zeros(2)
    mean_list = np.zeros((2, m))
    variance_list = np.zeros((2, m))
    for idx in range(2):
        sub = X_train[y_train == idx]
        prior_list[idx] = len(sub) / n
        mean_list[idx, :] = sub.mean(axis=0)
        variance_list[idx, :] = sub.var(axis=0)
    Xis = X_test
    pool = ThreadPool(5)
    preds = [pool.apply(max_class, args=(Xi, prior_list, mean_list, variance_list)) for Xi in Xis]
    return np.array(preds)


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def main():
    X, y, _ = load_data(standardize=False)
    accuracies = []
    start = time.time()
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, shuffle=True
        )
        preds = Parallel_NB(X_train, X_test, y_train)
        accuracies.append(accuracy(y_test, preds))
    end = time.time()
    print(f"Average accuracy: {np.mean(accuracies):.3f}")
    print(f"Training time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
