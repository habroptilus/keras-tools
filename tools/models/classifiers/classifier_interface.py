"""Keras Model Interface for classification."""
from tools.models.base import KerasNetInterface
from tools.plot import plot_history_classificaion
import numpy as np


class KerasClassifierInterface(KerasNetInterface):
    metrics = ["accuracy"]

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict_generator(self, generator):
        return np.argmax(self.model.predict_generator(generator), axis=1)

    def predict_proba_generator(self, generator):
        return self.model.predict_generator(generator)

    def plot(self, history):
        plot_history_classificaion(history)
