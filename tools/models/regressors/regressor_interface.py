from tools.models import KerasNetInterface


class KerasRegressorInterface(KerasNetInterface):
    metrics = ["mse"]

    def predict(self, X):
        return self.model.predict(X)

    def plot(self, history):s
        plot_history_regression(history)

