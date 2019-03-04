from typing import Dict
import logging as lg
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback
import json


class KerasNetInterface:
    original_layers: Dict = {}

    def __init__(self, trained_epochs, result_dir, batch_size, valid_rate, loss, optimizer):
        self.result_dir = result_dir
        self.batch_size = batch_size
        self.valid_rate = valid_rate
        self.loss = loss
        self.optimizer = optimizer
        model_path = result_dir / f"{self.create_flag()}/model_{trained_epochs:02d}.h5"
        history_path = result_dir / f"{self.create_flag()}/history_{trained_epochs:02d}.json"

        model_path.parent.mkdir(exist_ok=True, parents=True)
        if model_path.exists() and history_path.exists():
            self.trained_epochs = trained_epochs
            lg.info("Loading the trained model...")
            self.model = self.load_model(model_path)
            self.history = self.load_history(history_path)
            lg.info("Loaded.")
        else:
            lg.info("Not found such a trained model.")
            lg.info("Creating new model...")
            self.trained_epochs = 0
            self.history = {}
            self.model = self.construct()
            lg.info("Finished.")

    def load_model(self, model_path):
        return load_model(str(model_path), custom_objects=self.original_layers)

    def load_history(self, history_path):
        with history_path.open("r") as f:
            history = json.load(f)
        return history

    def fit(self, X, y, epochs):
        if self.trained_epochs >= epochs:
            lg.info(f"This model has already been traiend up to {self.trained_epochs} epochs")
            return
        callbacks = self.create_callbacks()
        self.model.fit(X, y, initial_epoch=self.trained_epochs, epochs=epochs,
                       batch_size=self.batch_size, callbacks=callbacks, validation_split=self.valid_rate)
        return self.history

    def save_model(self, save_path):
        self.model.save(str(save_path))

    def create_callbacks(self):
        model_path = self.result_dir / (self.create_flag()+"/model_{epoch:02d}.h5")
        mcp = ModelCheckpoint(str(model_path))
        hcp = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_history(epoch, logs))
        return [mcp, hcp]

    def save_history(self, epoch, logs):
        epoch += 1
        history_path = self.result_dir / f"{self.create_flag()}/history_{epoch:02d}.json"
        if len(self.history) == 0:
            self.history = {k: [v] for k, v in logs.items()}
        else:
            for k, v in logs.items():
                self.history[k].append(v)
        with history_path.open("w") as f:
            json.dump(self.history, f)
        self.trained_epochs = epoch

    def plot_history(self, epoch):
        history_path = self.result_dir / self.create_flag() / f"history_{epoch:02d}.json"
        if history_path.exists():
            with history_path.open("r") as f:
                history = json.load(f)
            self.plot(history)
        else:
            raise Exception(f"Not found {history_path}")

    def evaluate(self, X, y):
        scores = self.model.evaluate(X, y)
        metrics_names = self.model.metrics_names
        return {metrics_names[i]: scores[i] for i in range(len(scores))}

    def create_flag(self):
        base_flag = f"{self.batch_size}_{self.valid_rate}_{self.loss}_{self.optimizer}"
        return f"{self.model_flag()}_{base_flag}"

    def plot(self):
        pass

    def predict(self):
        pass

    def construct(self):
        pass

    def model_flag(self):
        pass
