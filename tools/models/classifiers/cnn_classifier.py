
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Model
from tools.models import KerasClassifierInterface


class CNNClassifier(KerasClassifierInterface):

    def __init__(self, result_dir, input_height, input_width, input_channels, filters=64, kernel_size=(3, 3),
                 trained_epochs=0, valid_rate=0.3, batch_size=256, pool_size=(2, 2), med_dim=128, output_dim=10,
                 dropout_rate1=0.25, dropout_rate2=0.5, activation="relu",
                 loss='categorical_crossentropy', optimizer='rmsprop'):

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size
        self.med_dim = med_dim
        self.output_dim = output_dim
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.activation = activation
        super().__init__(trained_epochs, result_dir, batch_size, valid_rate, loss, optimizer)

    def construct(self):
        inputs = Input(shape=(self.input_height, self.input_width, self.input_channels))
        x = Conv2D(self.filters, self.kernel_size, activation=self.activation)(inputs)
        x = MaxPooling2D(pool_size=self.pool_size)(x)
        x = Dropout(self.dropout_rate1)(x)
        x = Flatten()(x)
        x = Dense(self.med_dim, activation=self.activation)(x)
        x = Dropout(self.dropout_rate2)(x)
        predictions = Dense(self.output_dim, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def model_flag(self):
        return f"cnnc_{self.input_height}_{self.input_width}_{self.input_channels}_{self.filters}_"
        f"{self.pool_size}_{self.med_dim}_{self.output_dim}_{self.dropout_rate1}_{self.dropout_rate2}_{self.activation}"
