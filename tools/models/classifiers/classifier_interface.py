"""Keras Model Interface for classification."""
from tools.models import KerasNetInterface
from tools.plot import plot_history_classificaion
import numpy as np


class KerasClassifierInterface(KerasNetInterface):
    metrics = ["accuracy"]

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def plot(self, history):
        plot_history_classificaion(history)
