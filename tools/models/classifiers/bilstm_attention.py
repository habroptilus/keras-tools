from tools.models import KerasClassifierInterface
from keras.layers import Input, Dense, LSTM, Embedding
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from tools.layers import Attention


class BiLSTM_Attention(KerasClassifierInterface):
    """BiLSTMとAttentionを使ったKerasのModel class."""
    original_layers = {'Attention': Attention}

    def __init__(self, input_dim, seq_len, result_dir, batch_size=1, valid_rate=None, embedded_dim=200, dropout_rate=0.3, lstm_out=512,
                 trained_epochs=0, loss="categorical_crossentropy", optimizer="adam"):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.dropout_rate = dropout_rate
        self.lstm_out = lstm_out
        self.embedded_dim = embedded_dim
        self.loss = loss
        self.optimizer = optimizer
        super().__init__(trained_epochs, result_dir, batch_size, valid_rate)

    def construct(self):
        inputs = Input(shape=(self.seq_len,), dtype='int32')
        embedded = Embedding(output_dim=self.embedded_dim, input_dim=self.input_dim,
                             input_length=self.seq_len)(inputs)
        x = Bidirectional(
            LSTM(self.lstm_out, return_sequences=True, dropout=self.dropout_rate))(embedded)
        encoded = Attention(name="attention")(x)
        preds = Dense(2, activation='softmax')(encoded)
        model = Model(inputs=inputs, outputs=preds)
        model.compile(optimizer=Adam(), loss=self.loss, metrics=["accuracy"])
        return model

    def get_attention_vectors(self, X):
        attention_layer_model = Model(inputs=self.model.input,
                                      outputs=self.model.get_layer("attention").output)
        attention_output = attention_layer_model.predict(X)
        attention_vectors = attention_output.reshape(attention_output.shape[0],
                                                     attention_output.shape[-1])
        return attention_vectors

    def model_flag(self):
        return f"bla_{self.input_dim}_{self.seq_len}_{self.dropout_rate}_{self.lstm_out}_{self.embedded_dim}_{self.loss}_{self.optimizer}"
