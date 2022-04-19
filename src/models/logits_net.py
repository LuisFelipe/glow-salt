# -*- coding: utf-8 -*-
"""
module logits_net.py
-------------------------
    Auxiliary Network Layers used to calculate logits in the foward direction of the invertible 
    neural network.
    In `Glow` and `RealNVP` original models, 
    this network is just a global pooling plus a linear layer.
    For more details, refers to:
        - openAI official repository ´Glow´{https://github.com/openai/glow};
        - google brain official repository ´RealNVP´{https://github.com/tensorflow/models/tree/master/research/real_nvp};
        - `Glow` paper {https://arxiv.org/abs/1807.03039};
        - `RealNVP` paper {https://arxiv.org/abs/1605.08803};
        - `NICE` paper {https://arxiv.org/abs/1410.8516};

    Here we implement a more robust Network. 
""" 
import tensorflow as tf
from layers.zero_init import Conv2dTranspose
from flow.config import Config
layers = tf.keras.layers


class LogitsNet(layers.Layer):
    """ Logits Network Layers.
    A block of Nn layers that is used to calculate the forward logist of the invertible network.
    """
    def __init__(self, name=None, **kwargs):
        """ Network initialization.
        :param name: layers unique identifier name.
        :param kwargs: kwargs dict like.
        """
        super().__init__(name=name, **kwargs)
        self._init_config()

    def _init_config(self):
        _cfg = Config()
        self.n_levels = int(_cfg.get("model.levels"))
        self.n_filters = int(_cfg.get("MODEL.N_FILTERS", 64))

    def build(self, input_shape):
        if self.built:
            return

        self._layers = list()

        for i in range(self.n_levels):
            self._layers.append(
                Conv2dTranspose(
                    filters=self.n_filters,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    dropout_rate=-1.,
                    padding="same"
                )
            )

        self._layers.append(
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                use_bias=True,
                activation=tf.nn.sigmoid
            )
        )

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """Calls the layer on new inputs.

        In this case `call` just reapplies
        all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        :param inputs:
            A single input tensor ;

        :param kwargs:
            Dict like named parameters;

        :returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one output.
        """
        z = inputs
        logits = z

        for l in self._layers:
            logits = l(logits)
        # tf.add_to_collection("checkpoints", logits)
        return logits
