#!/usr/bin/env python3

from dataset import Dataset
from datetime import datetime
from transformer import create_transformer
import tensorflow as tf
import tensorboard


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dm):
        super(CustomSchedule, self).__init__()
        self.dm = dm
        self.dm = tf.cast(self.dm, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (4000 ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(transformer, dataset, epochs=5, verbose=True):
    """trains a transformer for translation"""
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    def loss_function(y, y_pred, sample_weight=None):
        y = tf.squeeze(y, axis=-1)
        loss_ = loss_object(y, y_pred)
        mask = tf.math.logical_not(tf.math.equal(y, 0))
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_, axis=-1) / tf.reduce_sum(mask, axis=-1)

    learning_rate = CustomSchedule(dm)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer.compile(optimizer=optimizer, loss=loss_function)
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    history = transformer.fit(dataset.data_train, epochs=epochs, verbose=verbose)
    return history


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.set_random_seed(0)
    
    dataset_name = 'para_crawl/enel'
    N = 4
    dm = 128
    h = 8
    hidden = 512
    max_len_input = 50
    max_len_target = 50
    epochs = 5
    batch_size = 32

    dataset = Dataset(
        dataset_name,
        batch_size,
        max_len_input,
        max_len_target,
        load_path='.')
    
    input_vocab = dataset.encoder_input.vocab_size + 2
    target_vocab = dataset.encoder_target.vocab_size + 2
    transformer = create_transformer(
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_len_input,
        max_len_target)
    transformer.summary()
    history = train_transformer(transformer, dataset)
    transformer.save('./entoel.h5')
