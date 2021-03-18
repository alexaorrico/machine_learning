#!/usr/bin/env python3
import tensorflow as tf
from train import train_transformer

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)
transformer = train_transformer('ted_hrlr_translate/pt_to_en', 4, 128, 8, 512, 32, 40, 2)
transformer.save('pt2en.h5')
