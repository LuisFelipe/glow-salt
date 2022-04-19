# -*- coding: utf-8 -*-
"""
module conv2d_zeros.py
-------------------------
    Conv2d_zeros layers used in the prior calculation.
    Code adapted from the official openAI ´Glow´{https://github.com/openai/glow} repository.
"""
import tensorflow as tf
layers = tf.keras.layers


class Conv2dZeros(layers.Layer):
    """Conv2d_zeros layers used in the prior calculation.
    Code adapted from the official openAI ´Glow´{https://github.com/openai/glow} repository.
    """

    def __init__(
            self, filters, kernel_size=[3, 3], strides=[1, 1],
            padding="SAME", logscale_factor=3, skip=1, name=None, **kwargs
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
        self.padding = padding.upper()
        self.logscale_factor = logscale_factor
        self.skip = skip
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return

        x_shape = input_shape.as_list()
        n_in = int(x_shape[3])
        filter_shape = self.kernel_size + [n_in, self.filters]
        self.w = self.add_weight(
            name=self.name + ".W",
            shape=filter_shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        self.logs = self.add_weight(
            name=self.name + ".logs",
            shape=[1, self.filters],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        self.b = self.add_weight(
            name=self.name + ".b",
            shape=[1, 1, 1, self.filters],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs

        stride_shape = [1] + self.strides + [1]
        if self.skip == 1:
            x = tf.nn.conv2d(
                input=x,
                filters=self.w,
                strides=stride_shape,
                padding=self.padding,
                data_format='NHWC'
            )
        else:
            assert self.strides[0] == 1 and self.strides[1] == 1
            x = tf.nn.atrous_conv2d(
                value=x,
                filters=self.w,
                rate=self.skip,
                padding=self.padding
            )
        x += self.b
        x *= tf.exp(self.logs * self.logscale_factor)
        return x
