# -*- coding: utf-8 -*-
"""
module actnorm.py
-----------------------------
Actnorm layer as described in `Glow: Generative Flow with invertible 1 x 1 Convolutions`
"""
import tensorflow as tf
from flow.config import Config
import utils
layers = tf.keras.layers



class ActNorm(layers.Layer):
    """An implementation of the ActNormLayer, this base class serves several
    utility functions used by their derivatives: Bias and Scale layer"""
    def __init__(self, name, logscale_factor=3., **kwargs):
        """
        Initialization Function.
        :param name: a custom name of the layer
        """
        self.built = False
        self.logscale_factor = logscale_factor
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if len(input_shape) == 3:
            x_shape, logdet_shape, _eps_shape = input_shape
        else:
            x_shape, logdet_shape = input_shape

        if self.built:
            return
        x_shape = x_shape.as_list()
        _shape = (1, 1, 1, x_shape[3])

        self.b = self.add_weight(
            self.name + ".b",
            shape=_shape,
            dtype=tf.float32,
            initializer=tf.initializers.zeros(),
            trainable=True,
        )

        self.logs = self.add_weight(
            self.name + ".logs",
            shape=_shape,
            dtype=tf.float32,
            initializer=tf.initializers.zeros(),
            trainable=True,
        )

        self.initialized = self.add_weight(
            self.name + ".initialized",
            shape=(),
            dtype=tf.bool,
            initializer=tf.initializers.zeros(),
            # initializer=False,
            trainable=False,
        )

        super().build(input_shape)
        self.built = True

    def data_dependent_init(self, strategy, *args):
        """
        Single replica data dependent layer initialization.
        :param args: positional args.
        """
        def init(t, *args):
            return t.assign(args[0])

        x = args[0]
        # merge x values
        all_x = strategy.experimental_local_results(x)
        x = tf.concat(all_x, axis=0)
        
        x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
        x_var = tf.reduce_mean(x ** 2, [0, 1, 2], keepdims=True)
        scale = tf.math.log(
            1. / (tf.sqrt(x_var) + 1e-6)
        ) / self.logscale_factor
        
        strategy.extended.update(
            self.b, init, args=(-x_mean,)
        )

        strategy.extended.update(
            self.logs, init, args=(scale,)
        )

        strategy.extended.update(
            self.initialized, init, args=(True,)
        )

    def forward(self, inputs, training=True):
        if len(inputs) == 3:
            x, logdet, _eps = inputs
        else:
            x, logdet = inputs
            _eps = None
            
        if not self.initialized:
            context = tf.distribute.get_replica_context()
            context.merge_call(self.data_dependent_init, args=(x,))
            # self.data_dependent_init(x)

        x_shape = x.shape.as_list()
        logdet_factor = int(x_shape[1]) * int(x_shape[2])
        logs = tf.maximum(self.logs, 1e-10)
        logs = logs * self.logscale_factor

        dlogdet = tf.reduce_sum(logs) * logdet_factor
        x += self.b
        x = x * tf.exp(logs)
        logdet += dlogdet

        if _eps is not None:
            return x, logdet, _eps
        else:
            return x, logdet

    def backward(self, inputs, training=True):
        if len(inputs) == 3:
            x, logdet, _eps = inputs
        else:
            x, logdet = inputs
            _eps = None
        x_shape = x.shape.as_list()
        logdet_factor = int(x_shape[1]) * int(x_shape[2])
        logs = tf.maximum(self.logs, 1e-10)
        logs = logs * self.logscale_factor

        dlogdet = -tf.reduce_sum(logs) * logdet_factor
        x = x * tf.exp(-logs)
        x -= self.b

        logdet += dlogdet

        if _eps is not None:
            return x, logdet, _eps
        else:
            return x, logdet

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
