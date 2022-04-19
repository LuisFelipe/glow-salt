# -*- coding: utf-8 -*-
"""
module squeeze.py
-----------------------------
Squeeze layer.`
"""
import tensorflow as tf
layers = tf.keras.layers


class Squeeze(layers.Layer):
    """Squeeze layer. pushes a factor os the input to the channel dimension."""
    def __init__(self, name, factor=2, **kwargs):
        """
        Initialization Function.
        :param name: a custom name of the layer
        """
        self.factor = factor
        super().__init__(name=name, **kwargs)

    def forward(self, inputs, training=True):

        if len(inputs) == 3:
            z, logdet, _eps = inputs
        else:
            z, logdet = inputs
            _eps = None

        if self.factor == 1:
            return z, logdet, _eps
        z = tf.nn.space_to_depth(z, self.factor)
        # z = self._squeeze(z)
        if _eps is not None:
            # _eps = self._squeeze(_eps)
            _eps = tf.nn.space_to_depth(_eps, self.factor)
            return z, logdet, _eps
        else:
            return z, logdet

    def backward(self, inputs, training=True):

        if len(inputs) == 3:
            z, logdet, _eps = inputs
        else:
            z, logdet = inputs
            _eps = None

        if self.factor == 1:
            if _eps is not None:
                return z, logdet, _eps
            else:
                return z, logdet

        # z = self._unsqueeze(z)
        z = tf.nn.depth_to_space(z, self.factor)
        if _eps is not None:
            # _eps = self._unsqueeze(_eps)
            _eps = tf.nn.depth_to_space(_eps, self.factor)
            return z, logdet, _eps
        else:
            return z, logdet

    def call(self, inputs, training=False, forward=True):
        """Calls the model on new inputs.

        In this case `call` just reapplies
        all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        :param inputs:
            A single input tensor ;
        :param training:
            Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
        :param forward:
            True if applying the forward pass.

        :returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        if forward:
            out = self.forward(inputs, training)
        else:
            out = self.backward(inputs, training)
        return out
