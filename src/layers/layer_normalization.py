# -*- coding: utf-8 -*-
"""
module nn.py
-----------------------------
Neural network block used inside the nvp layer for the operators S and T.
"""
import tensorflow as tf
layers = tf.keras.layers


class LayerNormalization(layers.Layer):
    """Layer Normalization implementation.
    Normalizes the input tensor x, averaging over the last dimension.
    """
    def __init__(self, epsilon=1e-20, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        if self.built:
            return
        x_shape = input_shape.as_list()
        self.scale = self.add_weight(
            name=self.name + "_scale",
            shape=[x_shape[-1]],
            initializer=tf.ones_initializer(),
            trainable=True
        )
        self.bias = self.add_weight(
            name=self.name + "_bias",
            shape=[x_shape[-1]],
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        super().build(input_shape)
        self.built = True

    def call(self, x, training=None):
        """Layer Normalization call.
        Normalizes the input tensor x, averaging over the last dimension.
        **Math:**
            ```
                norm_x = (x - mean(x, axis=-1)) * 1/sqrt(variance(x), axis=-1)
                norm_x = norm_x * scale * bias
            ```
        :param x: layer input tensor.
        :return: x normalized.
        """
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm_x = (x - mean)/tf.sqrt(variance + self.epsilon)
        return (norm_x * self.scale) + self.bias
