# -*- coding: utf-8 -*-
"""
module conv2d_transpose.py
-------------------------
    Conv2d_zeros layers used in the prior calculation.
    Code adapted from the official openAI ´Glow´{https://github.com/openai/glow} repository.
"""
import tensorflow as tf


class Conv2dTranspose(tf.keras.layers.Layer):
    """Conv2d_zeros layers used in the prior calculation.
    Code adapted from the official openAI ´Glow´{https://github.com/openai/glow} repository.
    """

    def __init__(
            self, filters, kernel_size=[3, 3], strides=[1, 1], dropout_rate=-1.0,
            padding="same", name=None, **kwargs
    ):
        """
        Conv2dZeros initializer.
        :param filters: number of filters to be applied. The output channels size.
        :param kernel_size: a list or tuple indicating the kernel size.
        :param strides: a list or tuple indicating the convolution strides.
        :param padding: 'VALID' or 'SAME'.
        :param name: layers name.
        :param kwargs: kwargs dict.
        """
        self.filters = filters
        self.kernel_size = list(kernel_size)
        self.strides = list(strides)
        self.padding = padding.lower()
        self.dropout_rate = dropout_rate
        super().__init__(name=name, **kwargs)
        self.init_sublayers()

    def init_sublayers(self):
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            self.filters, 
            self.kernel_size, 
            # strides=(1, 1), 
            strides=self.strides, 
            padding='same', 
            activation=None, 
            use_bias=True, 
            kernel_initializer='glorot_uniform', 
            bias_initializer='zeros', 
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding=self.padding,
            use_bias=True,
            activation=None
        )

        if self.dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(
                self.dropout_rate, 
                name=self.name + ".att_dropout"
            )

    def call(self, inputs, **kwargs):
        x = inputs

        out = self.conv_transpose(x)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.conv2(out)
        out_a, out_b = tf.split(out, 2, axis=-1)
        # gate multiply
        out = out_a * tf.nn.sigmoid(out_b)

        return tf.nn.relu(out)
