from keras.layers import Layer, Dropout, Dense
import keras.backend as K
from keras.initializers import Zeros, Ones
import math


class TransformerBasedEncoder(Layer):
    """TransformerのEncoder部分に相当するLayer.

    : positional encoding
    : dropout

    hopping_num回以下を繰り返す
    --------------------
    : Residual normalized  SelfAttetion
    : Residual normalized  FeedForwardNN
    --------------------
    : layer normalization

    # shape
    input_shape = output_shape = (batch_size, length, embedded_dim)
    """

    def __init__(self, hopping_num, head_num, dropout_rate, **kwargs):
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        super(TransformerBasedEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = Dropout(self.dropout_rate)

        self.attention_block_list = []
        for _ in range(self.hopping_num):
            attention_layer = SelfAttention(
                self.head_num, self.dropout_rate, name='self_attention')
            ffn_layer = FeedForwardNetwork(self.dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper(
                    attention_layer, self.dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(
                    ffn_layer, self.dropout_rate, name='ffn_wrapper'),
            ])
        self.output_normalization = LayerNormalization()

        super().build(input_shape)

    def call(self, x):
        embedded_input = self.add_position_embedding(x)
        query = self.input_dropout_layer(embedded_input)

        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with K.name_scope(f'hopping_{i}'):
                query = attention_layer(query)
                query = ffn_layer(query)
        # [batch_size, length, hidden_dim]
        return self.output_normalization(query)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate,
            "hopping_num": self.hopping_num,
            "head_num": self.head_num
        }
        base_config = super(TransformerBasedEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResidualNormalizationWrapper(Layer):
    def __init__(self, layer, dropout_rate, **kwargs):
        self.layer = layer
        self.dropout_rate = dropout_rate
        super(ResidualNormalizationWrapper, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layer_normalization = LayerNormalization()
        self.dropout_layer = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x):
        tensor = self.layer_normalization(x)
        tensor = self.layer(tensor)
        tensor = self.dropout_layer(tensor)
        return x + tensor

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNormalization(Layer):

    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.scale = self.add_weight('layer_norm_scale', shape=[hidden_dim],
                                     initializer=Ones())
        self.bias = self.add_weight('layer_norm_bias', [hidden_dim],
                                    initializer=Zeros())
        super().build(input_shape)

    def call(self, x, epsilon=1e-6):
        mean = K.mean(x, axis=[-1], keepdims=True)
        variance = K.mean(K.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * K.sqrt(variance + epsilon)

        return norm_x * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class FeedForwardNetwork(Layer):
    '''
    Transformer 用の Position-wise Feedforward Neural Network です。
    '''

    def __init__(self, dropout_rate, **kwargs):
        self.dropout_rate = dropout_rate
        super(FeedForwardNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        hidden_dim = input_shape[-1]
        self.filter_dense_layer = Dense(hidden_dim * 4, use_bias=True,
                                        activation="relu", name='filter_layer')
        self.output_dense_layer = Dense(
            hidden_dim, use_bias=True, name='output_layer')
        self.dropout_layer = Dropout(self.dropout_rate)

        super(FeedForwardNetwork, self).build(input_shape)

    def call(self, x):
        '''
        FeedForwardNetwork を適用します。
        :param input: shape = [batch_size, length, hidden_dim]
        :return: shape = [batch_size, length, hidden_dim]
        '''
        tensor = self.filter_dense_layer(x)
        tensor = self.dropout_layer(tensor)
        return self.output_dense_layer(tensor)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate
        }
        base_config = super(FeedForwardNetwork, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SelfAttention(Layer):
    """Multi head, scaled dot production,dropoutを含めたSelf Attetion layer."""

    def __init__(self, head_num, dropout_rate, **kwargs):
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.hidden_dim = input_shape[-1]

        self.q_dense_layer = Dense(
            self.hidden_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = Dense(
            self.hidden_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = Dense(
            self.hidden_dim, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = Dense(
            self.hidden_dim, use_bias=False, name='output_dense_layer')
        self.attention_dropout_layer = Dropout(self.dropout_rate)

        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        q = self.q_dense_layer(x)  # [batch_size, q_length, hidden_dim]
        k = self.k_dense_layer(x)  # [batch_size, m_length, hidden_dim]
        v = self.v_dense_layer(x)

        # [batch_size, head_num, q_length, hidden_dim/head_num]
        q = self._split_head(q)
        # [batch_size, head_num, m_length, hidden_dim/head_num]
        k = self._split_head(k)
        # [batch_size, head_num, m_length, hidden_dim/head_num]
        v = self._split_head(v)

        depth = self.hidden_dim // self.head_num
        q *= depth ** -0.5  # for scaled dot production

        # ここで q と k の内積を取ることで、query と key の関連度のようなものを計算します。

        k_t = K.permute_dimensions(k, (0, 1, 3, 2))

        # [batch_size, head_num, q_length, k_length]
        logit = K.batch_dot(q, k_t)

        # maskは一旦保留
        # logit += tf.to_float(attention_mask) * input.dtype.min  # mask は pad 部分などが1, 他は0

        # softmax を取ることで正規化します
        attention_weight = K.softmax(logit)
        attention_weight = self.attention_dropout_layer(attention_weight)

        # 重みに従って value から情報を引いてきます
        # [batch_size, head_num, q_length, hidden_dim/head_num]
        attention_output = K.batch_dot(attention_weight, v)
        # [batch_size, q_length, hidden_dim]
        attention_output = self._combine_head(attention_output)
        return self.output_dense_layer(attention_output)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _split_head(self, x):
        '''
        入力の tensor の hidden_dim の次元をいくつかのヘッドに分割します。
        入力 shape: [batch_size, length, hidden_dim] の時
        出力 shape: [batch_size, head_num, length, hidden_dim//head_num]
        となります。
        '''
        with K.name_scope('split_head'):
            batch_size, length, hidden_dim = K.int_shape(x)
            x = K.reshape(
                x, (K.shape(x)[0], length, self.head_num, self.hidden_dim // self.head_num))
            return K.permute_dimensions(x, [0, 2, 1, 3])

    def _combine_head(self, x):
        '''
        入力の tensor の各ヘッドを結合します。 _split_head の逆変換です。
        入力 shape: [batch_size, head_num, length, hidden_dim//head_num] の時
        出力 shape: [batch_size, length, hidden_dim]
        となります。
        '''
        with K.name_scope('combine_head'):
            batch_size, _, length, _ = K.int_shape(x)
            x = K.permute_dimensions(x, [0, 2, 1, 3])
            return K.reshape(x, (K.shape(x)[0], length, self.hidden_dim))

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate,
            'head_num': self.head_num
        }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AddPositionalEncoding(Layer):
    """Positional encoding を追加するレイヤー"""

    def __init__(self, **kwargs):
        super(AddPositionalEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        super(AddPositionalEncoding, self).build(input_shape)

    def call(self, x, mask=None):
        fl_type = x.dtype
        _, max_length, depth = K.int_shape(x)

        depth_counter = K.arange(depth) // 2 * 2  # 0, 0, 2, 2, 4, ...
        depth_matrix = K.tile(K.expand_dims(depth_counter, 0), [
                              max_length, 1])  # [max_length, depth]
        depth_matrix = K.pow(10000.0, K.cast(
            depth_matrix / depth, fl_type))  # [max_length, depth]

        # cos(x) == sin(x + π/2)
        # 0, π/2, 0, π/2, ...
        phase = K.cast(K.arange(depth) % 2, fl_type) * math.pi / 2
        phase_matrix = K.tile(K.expand_dims(phase, 0), [
                              max_length, 1])  # [max_length, depth]

        pos_counter = K.arange(max_length)
        pos_matrix = K.cast(K.tile(K.expand_dims(pos_counter, 1), [
                            1, depth]), fl_type)  # [max_length, depth]

        positional_encoding = K.sin(pos_matrix / depth_matrix + phase_matrix)
        # [batch_size, max_length, depth]
        positional_encoding = K.tile(K.expand_dims(
            positional_encoding, 0), [K.shape(x)[0], 1, 1])
        return x + positional_encoding

    def compute_output_shape(self, input_shape):
        return input_shape


class SingleHeadSelfAttention(Layer):
    """single head self attention layer (scaled dot production,dropout含む)"""

    def __init__(self, dropout_rate, **kwargs):
        self.dropout_rate = dropout_rate
        super(SingleHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.depth = input_shape[-1]
        self.W_q = self.add_weight(name='W_q', shape=(
            self.depth, self.depth), initializer='normal', trainable=True)
        self.W_k = self.add_weight(name='W_k', shape=(
            self.depth, self.depth), initializer='normal', trainable=True)
        self.W_v = self.add_weight(name='W_v', shape=(
            self.depth, self.depth), initializer='normal', trainable=True)
        self.W_o = self.add_weight(name='W_o', shape=(
            self.depth, self.depth), initializer='normal', trainable=True)
        super(SingleHeadSelfAttention, self).build(input_shape)

    def call(self, x, mask=None):
        q = K.dot(x, self.W_q)
        k = K.dot(x, self.W_k)
        v = K.dot(x, self.W_v)

        k_t = K.permute_dimensions(k, (0, 2, 1))

        q *= self.depth ** (-0.5)  # scaled dot-product

        logit = K.batch_dot(q, k_t)  # [batch_size, q_length, k_length]

        # softmax を取ることで正規化します
        attention_weight = K.softmax(logit)

        # dropout
        attention_weight = K.dropout(attention_weight, level=self.dropout_rate)

        # 重みに従って value から情報を引いてきます
        # [batch_size, q_length, depth]
        attention_output = K.batch_dot(attention_weight, v)
        output = K.dot(attention_output, self.W_o)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(SingleHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
