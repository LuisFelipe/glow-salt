# -*- coding: utf-8 -*-
"""
module linear_zeros.py
-------------------------
    Linear_zeros layers used in the prior calculation.
    Code adapted from the official openAI ´Glow´{https://github.com/openai/glow} repository.
"""
import tensorflow as tf
layers = tf.keras.layers

class LinearZeros(layers.Layer):
    """Linear_zeros layers used in the prior calculation.
    Code adapted from the official openAI ´Glow´{https://github.com/openai/glow} repository.
    """

    def __init__(
            self,
            units, logscale_factor=3, weight_norm=False,
            name=None, **kwargs
    ):
        """
        LinearZeros initializer.
        :param units: number of filters to be applied. The output channels size.
        :param name: layers name.
        :param kwargs: kwargs dict.
        """
        self.units = units
        self.logscale_factor = logscale_factor
        self.weight_norm = weight_norm
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return

        x_shape = input_shape.as_list()
        n_in = int(x_shape[1])
        self.w = self.add_weight(
            name=self.name + ".W",
            shape=[n_in, self.units],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        self.logs = self.add_weight(
            name=self.name + ".logs",
            shape=[1, self.units],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        self.b = self.add_weight(
            name=self.name + ".b",
            shape=[1, self.units],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs
        w = self.w
        logs = self.logs
        b = self.b
        if self.weight_norm:
            w = tf.nn.l2_normalize(self.w)
            logs = tf.nn.l2_normalize(self.logs)
            b = tf.nn.l2_normalize(self.b)

        x = tf.matmul(x, w)
        x += b
        x *= tf.exp(logs * self.logscale_factor)
        return x
