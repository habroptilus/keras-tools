from tools.models import KerasNetInterface
from tools.plot import plot_history_regression


class KerasRegressorInterface(KerasNetInterface):
    metrics = ["mse"]

    def predict(self, X):
        return self.model.predict(X)

    def plot(self, history):
        plot_history_regression(history)
