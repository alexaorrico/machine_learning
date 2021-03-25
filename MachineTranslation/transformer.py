#!/usr/bin/env python3
"""Transformer Model and all subcomponents"""
import numpy as np
import tensorflow as tf


def create_transformer(num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_max_input, pe_max_target, rate=0.1):
        I1 = tf.keras.Input(shape=(49,))
        I2 = tf.keras.Input(shape=(47,))
        x, mask = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_max_input, rate)(I1)
        y = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_max_target, rate)([I2, x, mask])
        return TransformerModel(inputs=[I1, I2], outputs=y)

class TransformerModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, name='encoderLayer', **kwargs):
        super(EncoderLayer, self).__init__(name=name, **kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff1 = tf.keras.layers.Dense(dff, activation='relu')
        self.ff2 = tf.keras.layers.Dense(d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        x, mask = inputs
        attn_output = self.mha([x, x, x, mask])
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ff_out = self.ff1(out1)
        ff_out = self.ff2(ff_out)
        ff_out = self.dropout2(ff_out, training=training)
        out2 = self.layernorm2(out1 + ff_out)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, name='decoderLayer', **kwargs):
        super(DecoderLayer, self).__init__(name=name, **kwargs)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ff1 = tf.keras.layers.Dense(dff, activation='relu')
        self.ff2 = tf.keras.layers.Dense(d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        x, enc_output, look_ahead_mask, padding_mask = inputs
        attn1 = self.mha1([x, x, x, look_ahead_mask])
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2([out1, enc_output, enc_output, padding_mask])
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ff_out = self.ff1(out2)
        ff_out = self.ff2(ff_out)
        ff_out = self.dropout3(ff_out, training=training)
        out3 = self.layernorm3(ff_out + out2)

        return out3


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 pe_max, rate=0.1, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(pe_max,
                                                self.d_model)
        self.mask = PaddingMask()
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]

        mask = self.mask(inputs)
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i]([x, mask], training=training)

        return x, mask


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 pe_max, rate=0.1, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            pe_max, d_model)
        self.padding_mask = PaddingMask()
        self.look_ahead_mask = LookAheadMask()
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        x, enc_output, dec_padding_mask = inputs
        seq_len = tf.shape(x)[1]
        tar_padding_mask = self.padding_mask(x)
        look_ahead_mask = self.look_ahead_mask(tar_padding_mask)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i]([x, enc_output, look_ahead_mask, dec_padding_mask], training=training)
        return self.final_layer(x)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name='multiHeadAttention', **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output

class PaddingMask(tf.keras.layers.Layer):
    def __init__(self, name='paddingMask', **kwargs):
        super(PaddingMask, self).__init__(name=name, **kwargs)
    
    def call(self, sequence):
        padding = tf.cast(tf.math.equal(sequence, 0), tf.float32)
        padding = padding[:, tf.newaxis, tf.newaxis, :]
        return padding

class LookAheadMask(tf.keras.layers.Layer):
    def __init__(self, name='lookAheadMask', **kwargs):
        super(LookAheadMask, self).__init__(name=name, **kwargs)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        look_ahead = 1 - tf.linalg.band_part(tf.fill([seq_len, seq_len], 1.), -1, 0)
        look_ahead = tf.maximum(inputs, look_ahead)
        return look_ahead
         

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
