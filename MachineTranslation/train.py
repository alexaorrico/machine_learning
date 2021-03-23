#!/usr/bin/env python3

from dataset import Dataset
from transformer import create_transformer
import tensorflow as tf


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

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    train_step_signature = [
        tf.TensorSpec(
            shape=(
                None, None), dtype=tf.int64), tf.TensorSpec(
            shape=(
                None, None), dtype=tf.int64)]
    learning_rate = CustomSchedule(dm)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, targets):
        target_inputs = targets[:, :-1]
        target_real = targets[:, 1:]

        encoder_padding_mask, combined_mask, decoder_padding_mask = create_masks(
            inputs, target_inputs)

        with tf.GradientTape() as tape:
            predictions = transformer([inputs, target_inputs],
                                      training=True)
            loss = loss_function(target_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(target_real, predictions)

    for epoch in range(epochs):

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inputs, targets)) in enumerate(data.data_train):
            train_step(inputs, targets)
            if verbose and not batch % 50:
                print(
                    'Epoch {}, batch {}: loss {} accuracy {}'.format(
                        epoch + 1,
                        batch,
                        train_loss.result(),
                        train_accuracy.result()))
        if verbose:
            print(
                'Epoch {}: loss {} accuracy {}'.format(
                    epoch + 1,
                    train_loss.result(),
                    train_accuracy.result()))

    return transformer


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    print(tf.executing_eagerly())
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

    data = Dataset(
        dataset_name,
        batch_size,
        max_len_input,
        max_len_target,
        load_path='.')
    
    input_vocab = data.encoder_input.vocab_size + 2
    target_vocab = data.encoder_target.vocab_size + 2
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
    tf.saved_model.save(transformer, './entoel/')
