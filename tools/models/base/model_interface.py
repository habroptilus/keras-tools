from typing import Dict
import logging as lg
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback
import json


class KerasNetInterface:
    original_layers: Dict = {}
    metrics = []

    def __init__(self, trained_epochs, result_dir, loss, optimizer):
        self.result_dir = result_dir
        self.loss = loss
        self.optimizer = optimizer
        model_path = result_dir / \
            f"{self.create_flag()}/model_{trained_epochs:02d}.h5"
        history_path = result_dir / \
            f"{self.create_flag()}/history_{trained_epochs:02d}.json"

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

    def fit(self, X_train, y_train, epochs, batch_size=1, valid_rate=None, X_valid=None, y_valid=None):
        if self.trained_epochs >= epochs:
            lg.info(
                f"This model has already been traiend up to {self.trained_epochs} epochs")
            return
        callbacks = self.create_callbacks()
        self.model.fit(X_train, y_train, initial_epoch=self.trained_epochs, epochs=epochs,
                       batch_size=batch_size, callbacks=callbacks, validation_data=(X_valid, y_valid), validation_split=valid_rate)
        return self.history

    def fit_generator(self, generator, epochs, valid_generator=None):
        if self.trained_epochs >= epochs:
            lg.info(
                f"This model has already been traiend up to {self.trained_epochs} epochs")
            return
        callbacks = self.create_callbacks()
        v_steps = len(valid_generator) if valid_generator is not None else None
        self.model.fit_generator(generator, epochs=epochs, steps_per_epoch=len(generator), initial_epoch=self.trained_epochs,
                                 validation_data=valid_generator, validation_steps=v_steps, callbacks=callbacks)

    def save_model(self, save_path):
        self.model.save(str(save_path))

    def create_callbacks(self):
        model_path = self.result_dir / \
            (self.create_flag() + "/model_{epoch:02d}.h5")
        mcp = ModelCheckpoint(str(model_path))
        hcp = LambdaCallback(on_epoch_end=lambda epoch,
                             logs: self.save_history(epoch, logs))
        return [mcp, hcp]

    def save_history(self, epoch, logs):
        epoch += 1
        history_path = self.result_dir / \
            f"{self.create_flag()}/history_{epoch:02d}.json"
        if len(self.history) == 0:
            self.history = {k: [v] for k, v in logs.items()}
        else:
            for k, v in logs.items():
                self.history[k].append(v)
        with history_path.open("w") as f:
            json.dump(self.history, f)
        self.trained_epochs = epoch

    def plot_history(self, epoch):
        history_path = self.result_dir / \
            self.create_flag() / f"history_{epoch:02d}.json"
        if history_path.exists():
            with history_path.open("r") as f:
                history = json.load(f)
            self.plot(history)
        else:
            raise Exception(f"Not found {history_path}")

    def evaluate(self, X, y):
        scores = self.model.evaluate(X, y)
        metrics_names = self.model.metrics_names
        if len(metrics_names) == 1:
            return {metrics_names[0]: scores}
        else:
            return {metrics_names[i]: scores[i] for i in range(len(scores))}

    def create_flag(self):
        pass

    def plot(self, history):
        pass

    def predict(self):
        pass

    def construct(self):
        pass
