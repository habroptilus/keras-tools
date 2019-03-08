from tools.models import KerasClassifierInterface
from tools.models.layers import MyAttention, TransformerBasedEncoder
from keras.layers import Input, Embedding, Dense
from keras.models import Model


class TransformerBasedModel(KerasClassifierInterface):
    """Transformerをベースに実装したClassifier."""

    original_layers = {'MyAttention': MyAttention,
                       "TransformerBasedEncoder": TransformerBasedEncoder}

    def __init__(self, input_dim, seq_len, result_dir, output_dim, hopping_num=1, head_num=3, embedded_dim=150, dropout_rate=0.3, lstm_out=512,
                 trained_epochs=0, loss="categorical_crossentropy", optimizer="adam"):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.dropout_rate = dropout_rate
        self.lstm_out = lstm_out
        self.embedded_dim = embedded_dim
        self.loss = loss
        self.optimizer = optimizer
        self.head_num = head_num
        self.hopping_num = hopping_num
        self.output_dim = output_dim
        super().__init__(trained_epochs, result_dir)

    def construct(self):
        inputs = Input(shape=(self.seq_len,), dtype='int32')
        embedded = Embedding(output_dim=self.embedded_dim, input_dim=self.input_dim,
                             input_length=self.seq_len)(inputs)
        encoded = TransformerBasedEncoder(
            hopping_num=self.hopping_num, head_num=self.head_num, dropout_rate=self.dropout_rate)(embedded)
        compressed = MyAttention(name="attention")(encoded)
        preds = Dense(2, activation='softmax')(compressed)
        model = Model(inputs=inputs, outputs=preds)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=self.metrics)
        return model

    def create_flag(self):
        return f"tfb_{self.input_dim}_{self.seq_len}_{self.dropout_rate}_{self.lstm_out}_{self.head_num}_{self.hopping_num}_{self.embedded_dim}_{self.output_dim}_{self.loss}_{self.optimizer}"
