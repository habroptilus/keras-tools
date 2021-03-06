from keras.engine.topology import Layer
import keras.backend as K


class Attention(Layer):
    """自作のattention layer."""

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], 1),
                                 initializer='normal',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        eij = K.squeeze(eij, axis=2)
        ai = K.exp(eij)
        Sum = K.expand_dims(K.sum(ai, axis=1), axis=1)
        weights = ai / Sum
        weights = K.expand_dims(weights, axis=1)
        weighted_input = K.batch_dot(weights, x)
        weighted_input = K.squeeze(weighted_input, axis=1)
        return weighted_input

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
