#!/usr/bin/env python3
"""Contains the Dataset class"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """Class to prepare the TF dataset"""

    def __init__(
            self,
            dataset_name=None,
            batch_size=32,
            max_len_input=50,
            max_len_target=50,
            load_path=None,
            save_path=None):
        """Initialize the Dataset"""
        self.data_train, info = tfds.load(
            dataset_name, split='train', as_supervised=True, with_info=True)
        self.encoder_input, self.encoder_target = self.get_encoders(
            self.data_train, load_path, save_path)
        self.data_train = self.data_train.map(self.tf_encode)
        def filter_len(inputs, targets):
            return tf.logical_and(
                tf.size(inputs) <= max_len_input,
                tf.size(targets) <= max_len_target)
        self.data_train = self.data_train.filter(filter_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(2**12, reshuffle_each_iteration=True).padded_batch(batch_size, ([None], [None]))
        self.data_train = self.data_train.map(lambda x, y: ({'input_1': x, 'input_2': y[:, :-1]}, y[:, 1:, tf.newaxis]))
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

    def get_encoders(self, data, load_path=None, save_path=None):
        """load or create the encoders"""
        if load_path:
            print('loading encoders')
            inputs = tfds.deprecated.text.SubwordTextEncoder.load_from_file(load_path + '/input_encoder')
            targets = tfds.deprecated.text.SubwordTextEncoder.load_from_file(load_path + '/target_encoder')
            print('encoders loaded')
        else:
            print('creating input encoder')
            inputs = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (inputs.numpy() for inputs, targets in data), 2**20)
            print('creating target encoder')
            targets = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (targets.numpy() for inputs, targets in data), 2**20)
            print('encoders created')
        if save_path:
            print('saving encoders')
            inputs.save_to_file(save_path + '/input_encoder')
            targets.save_to_file(save_path + '/target_encoder')
            print('encoders saved')
        return inputs, targets

    def encode(self, inputs, targets):
        """convert the text to tokens"""
        input_size = self.encoder_input.vocab_size
        target_size = self.encoder_target.vocab_size
        input_tokens = self.encoder_input.encode(inputs.numpy())
        target_tokens = self.encoder_target.encode(targets.numpy())
        input_tokens = [input_size] + \
            input_tokens + [input_size + 1]
        target_tokens = [target_size] + \
            target_tokens + [target_size + 1]
        return input_tokens, target_tokens

    def tf_encode(self, inputs, targets):
        """tensorflow wrapper for encode"""
        input_tokens, target_tokens = tf.py_function(
            self.encode, [inputs, targets], [tf.int64, tf.int64])
        input_tokens.set_shape([None])
        target_tokens.set_shape([None])
        return input_tokens, target_tokens
