#!/usr/bin/env python3

from masks import create_masks
from dataset import Dataset
from transformer import Transformer
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

def train_transformer(dataset_name, N, dm, h, hidden, max_len, batch_size, epochs):

    data = Dataset(dataset_name, batch_size, max_len)
    pt_vocab = data.tokenizer_pt.vocab_size + 2
    en_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden, pt_vocab, en_vocab, max_len, max_len)
    print(type(transformer))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
    learning_rate = CustomSchedule(dm)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                         epsilon=1e-9)
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            predictions = transformer(inp, tar_inp, 
                                        True, 
                                        enc_padding_mask, 
                                        combined_mask, 
                                        dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(tar_real, predictions)
    
    for epoch in range(epochs):
  
        train_loss.reset_states()
        train_accuracy.reset_states()
  
        for (batch, (inp, tar)) in enumerate(data.data_train):
            train_step(inp, tar)
            if batch % 50 == 0:
                print ('Epoch {}, batch {}: loss {} accuracy {}'.format(
                       epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    
        print ('Epoch {}: loss {} accuracy {}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

    return transformer
