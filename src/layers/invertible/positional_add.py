# -*- coding: utf-8 -*-
"""
module positional_add.py
-----------------------------
Positional embedding layer working as an additive invertible layer.
"""
import tensorflow as tf
layers = tf.keras.layers


class PositionalAdd(layers.Layer):
    """
    Positional embedding layer working as an additive invertible layer.
    """

    def __init__(
            self,
            name=None,
            **kwargs
    ):
        """
        Implementation of the positional embedding as a additive layer (see RealNVP or NICE) paper
        for more details on additive layers.
        """
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        """Creates the variables of the layer."""
        if len(input_shape) == 3:
            z_shape, logdet_shape, eps_shape = input_shape
        else:
            z_shape, logdet_shape = input_shape

        if not self.built:
            z_shape = z_shape.as_list()

            pos_shape = [z_shape[1], z_shape[2], z_shape[3]]
            # positional embedding
            self.pos_emb = self.add_weight(
                name=self.name + ".positional_emb",
                shape=pos_shape,
                dtype=tf.float32,
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                trainable=True
            )
            
            super().build(input_shape)
            self.built = True

    def forward(self, inputs):
        if len(inputs) == 3:
            z, logdet, _eps = inputs
        else:
            z, logdet = inputs
            _eps = None

        z += self.pos_emb

        if _eps is not None:
            return z, logdet, _eps
        else:
            return z, logdet

    def backwards(self, inputs):
        if len(inputs) == 3:
            z, logdet, _eps = inputs
        else:
            z, logdet = inputs
            _eps = None

        z -= self.pos_emb

        if _eps is not None:
            return [z, logdet, _eps]
        else:
            return [z, logdet]

    def call(self, inputs, training=None, forward=True):
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
            out = self.forward(inputs)
        else:
            out = self.backwards(inputs)

        return out
