import numpy as np
import time
from sklearn.model_selection import train_test_split
from .utils import load_data

class LogRegGradAscent:
    """Logistic regression using gradient ascent as defined in the notebook."""
    def __init__(self, lr=0.01, num_iterations=100):
        self.lr = lr
        self.num_iterations = num_iterations
        self.eps = 1e-10
        self.weights = None
        self.lls = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _log_likelihood(self, y, pred):
        pred = np.clip(pred, self.eps, 1 - self.eps)
        ll = y * np.log(pred) + (1 - y) * np.log(1 - pred)
        return ll.mean()

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.num_iterations):
            z = X @ self.weights
            pred = self._sigmoid(z)
            grad = np.mean((y - pred)[:, None] * X, axis=0)
            self.weights += self.lr * grad
            self.lls.append(self._log_likelihood(y, pred))

    def predict(self, X, threshold=0.5):
        prob = self._sigmoid(X @ self.weights)
        return (prob > threshold).astype(int)


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def main():
    X, y, _ = load_data(standardize=True)
    accuracies = []
    start = time.time()
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )
        model = LogRegGradAscent()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracies.append(accuracy(y_test, pred))
    end = time.time()
    print(f"Average accuracy: {np.mean(accuracies):.3f}")
    print(f"Training time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
