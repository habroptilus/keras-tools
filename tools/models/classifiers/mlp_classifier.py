"""MLP for Classification."""
from keras.layers import Input, Dense
from keras.models import Model
from .classifier_interface import KerasClassifierInterface


class MLPClassifier(KerasClassifierInterface):

    def __init__(self, result_dir, input_dim, output_dim, trained_epochs=0, batch_size=1, valid_rate=None,
                 med1_dim=300, med2_dim=100, activation="relu",
                 loss='categorical_crossentropy', optimizer='rmsprop'):

        self.input_dim = input_dim
        self.med1_dim = med1_dim
        self.med2_dim = med2_dim
        self.output_dim = output_dim
        self.activation = activation
        super().__init__(trained_epochs, result_dir, loss, optimizer)

    def construct(self):
        inputs = Input(shape=(self.input_dim,))
        x = Dense(self.med1_dim, activation=self.activation)(inputs)
        x = Dense(self.med2_dim, activation=self.activation)(x)
        predictions = Dense(self.output_dim, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=self.metrics)
        return model

    def create_flag(self):
        return f"mlpc_{self.input_dim}_{self.med1_dim}_{self.med2_dim}_{self.output_dim}_{self.activation}"
