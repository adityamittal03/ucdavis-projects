import numpy as np
import time
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import multiprocessing
from .utils import load_data


def euclidean(row1, row2):
    return np.sqrt(((row1 - row2) ** 2).sum())


def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for idx, row in enumerate(train):
        dist = euclidean(test_row, row)
        distances.append((dist, idx))
    distances.sort(key=lambda x: x[0])
    return [idx for _, idx in distances[:num_neighbors]]


def predict_classification(X_train, y_train, test_row, num_neighbors):
    neighbor_idx = get_neighbors(X_train, test_row, num_neighbors)
    output_values = y_train[neighbor_idx]
    return np.bincount(output_values).argmax()


def accuracy_metric(y_true, y_pred):
    return (y_true == y_pred).mean()


def main():
    X, y, _ = load_data(standardize=True)
    accuracies = []
    start = time.time()
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )
        inputs = range(len(X_test))
        num_cores = multiprocessing.cpu_count()
        def runner(t):
            return predict_classification(X_train, y_train, X_test[t], 25)
        preds = Parallel(n_jobs=num_cores)(delayed(runner)(t) for t in inputs)
        accuracies.append(accuracy_metric(y_test, np.array(preds)))
    end = time.time()
    print(f"Average accuracy: {np.mean(accuracies):.3f}")
    print(f"Training time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
