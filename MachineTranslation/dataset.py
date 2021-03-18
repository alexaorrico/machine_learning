#!/usr/bin/env python3
"""Contains the Dataset class"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """Class to prepare the dataset"""
    def __init__(self, dataset_name, batch_size, max_len):
        """Initialize the Dataset"""
        self.data_train, info = tfds.load(dataset_name, split='train', as_supervised=True, with_info=True)
        self.data_valid = tfds.load(dataset_name, split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)
        def filter_len(pt, en):
            return tf.logical_and(tf.size(pt) <= max_len,  tf.size(en) <= max_len)
        self.data_train = self.data_train.filter(filter_len)
        self.data_valid = self.data_valid.filter(filter_len)
        self.data_train = self.data_train.cache()
        train_examples = info.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(train_examples).padded_batch(batch_size, ([None], [None]))
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)
        self.data_valid = self.data_valid.padded_batch(batch_size, ([None], [None]))

    def tokenize_dataset(self, data):
        """create tokenizers"""
        pt = tfds.features.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in data), 2**15)
        en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in data), 2**15)
        return pt, en

    def encode(self, pt, en):
        """convert to tokens"""
        pt_size = self.tokenizer_pt.vocab_size
        en_size = self.tokenizer_en.vocab_size
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())
        pt_tokens = [pt_size] + pt_tokens + [pt_size + 1]
        en_tokens = [en_size] + en_tokens + [en_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """tensorflow wrapper"""
        pt_tokens, en_tokens = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
