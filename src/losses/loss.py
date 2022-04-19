# -*- coding: utf-8 -*-
"""
module loss.py
------------------------- 
    Layers that calculates the prior probability of a bit.
    Code adapted from the official openAI ´Glow´{https://github.com/openai/glow} repository.
"""
import tensorflow as tf
import numpy as np
from flow.config import Config
layers = tf.keras.layers


class Loss(layers.Layer):
    def __init__(self, weight_y=0., name=None, **kwargs):
        self.weight_y = weight_y
        super().__init__(name=name, **kwargs)
        self._init_config()

    def _init_config(self):
        _cfg = Config()
        self.global_batch_size = float(_cfg.get("flow.global_batch_size"))

    def build(self, input_shape):
        if self.built:
            return

        self.weight_y = tf.constant(
            self.weight_y, 
            dtype=tf.float32, 
            name="y_weight"
        )
        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x, y_mask, logdet, z, _eps, y_logits, prior_mean, prior_logstd = inputs
        # _prior= prior log density of x
        _prior = self.logp(prior_mean, prior_logstd, z)
        logdet += _prior
        # generative_loss
        bits_x = self.generative_loss(x, logdet)
        bits_y = self.predictive_loss(y_logits, y_mask)
        
        bits_x = tf.nn.compute_average_loss(
            bits_x, 
            global_batch_size=self.global_batch_size
        )

        bits_y = tf.nn.compute_average_loss(
            bits_y, 
            global_batch_size=self.global_batch_size
        )

        _loss = bits_x + bits_y * self.weight_y
        # _loss = tf.nn.compute_average_loss(
        #     bits_x + bits_y * self.weight_y,
        #     global_batch_size=self.global_batch_size
        # )

        _loss._name = "loss"
        bits_x._name = "bits_x"
        bits_y._name = "bits_y"

        return [_loss, bits_x, bits_y]

    def logp(self, mean, log_std, z):
        log_ps = -0.5 * (
                np.log(2. * np.pi) +
                (2. * log_std) + ((z - mean) ** 2.) /
                tf.exp(2. * log_std)
        )
        return tf.reduce_sum(log_ps, axis=[1, 2, 3])

    def generative_loss(self, x, logdet):
        # Generative loss
        x_shape = x.shape.as_list()
        nobj = -logdet
        # bits per subpixel
        bits_x = (
            nobj /
            (
                np.log(2.) * x_shape[1] *
                x_shape[2] * x_shape[3]
            )
        )
        return bits_x

    def predictive_loss(self, y_logits, y_mask):
        bits_y = tf.keras.backend.binary_crossentropy(
            y_mask[:, :, :, 0],
            y_logits[:, :, :, 0],
            from_logits=False
        )  # / np.log(2.)
        return tf.reduce_mean(bits_y, axis=[1, 2])
