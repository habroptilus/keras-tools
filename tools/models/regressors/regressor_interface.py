from tools.models import KerasNetInterface
from tools.plot import plot_history_regression


class KerasRegressorInterface(KerasNetInterface):

    def predict(self, X):
        return self.model.predict(X)

    def predict_generator(self, generator):
        return self.model.predict_generator(generator)

    def plot(self, history):
        plot_history_regression(history)
