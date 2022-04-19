# -*- coding: utf-8 -*-
"""
module prior.py
-------------------------
    Layers that calculates the prior probability distribution hyperparameters
    (the mean and log standard deviation) of a bit.
    Code adapted from the official openAI ´Glow´{https://github.com/openai/glow} repository.
"""
import tensorflow as tf
from layers import Conv2dZeros
from layers.self_attention import SelfAttention
from flow.config import Config
layers = tf.keras.layers


class Prior(layers.Layer):
    """
    Calculates the Mean and Variance of the prior probability distribution.
    """
    def __init__(self, name=None, **kwargs):
        """
        Prior initialization function.
        :param spatial_prior: if it should calculate the prior spatially (not only globally).
        :param y_cond: if it should condition the the prior probability on the output class.
        :param name: layers unique identifier name.
        :param kwargs: kwargs dict like.
        """
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return

        y_onehot, z = input_shape

        self.prior_net = PriorNet(
            z.as_list(),
            name=self.name + ".prior_net"
        )

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        [y_mask, z] = inputs
        z_shape = z.shape.as_list()
        nz = z_shape[-1]

        h = self.prior_net(y_mask)

        mean = h[:, :, :, :nz]
        log_std = h[:, :, :, nz:]

        return mean, log_std


class PriorNet(layers.Layer):
    """ Prior network layer.
    """
    def __init__(self, z_shape, name=None, **kwargs):
        """
        Prior network initialization.
        :param name: layers unique identifier name.
        :param kwargs: kwargs dict like.
        """
        super().__init__(name=name, **kwargs)
        self.z_shape = z_shape
        self._init_config()

    def _init_config(self):
        _cfg = Config()
        self.n_levels = int(_cfg.get("model.levels"))

    def build(self, input_shape):
        if self.built:
            return

        z_shape = self.z_shape[1:]
        nz = z_shape[-1]

        self.pos_emb = self.add_weight(
            name=self.name + ".positional_emb",
            shape=[z_shape[0], z_shape[1], z_shape[2]],
            dtype=tf.float32,
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            trainable=True
        )
        self.layers = list()
        self.r_layers = list()

        for i in range(self.n_levels):
            self.layers.append(
                tf.keras.layers.Conv2D(
                    filters=int(nz),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    use_bias=True,
                    activation=tf.nn.relu
                )

            )
            self.layers.append(
                tf.keras.layers.Conv2D(
                    filters=int(nz),
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                    use_bias=True,
                    activation=tf.nn.relu
                )

            )
        self.layers.append(
            # tf.keras.layers.Lambda(function=lambda x: x + self.pos_emb)
            lambda x: x + self.pos_emb
        )
        for i in range(1):
            self.layers.append(
                SelfAttention(
                    qkv_units=int(nz),
                    kernel_size=(1, 1),
                    num_heads=4,
                    dropout_rate=0.1,
                    normalize=True,
                    gatted=False,
                    # out_units=int(2 * nz),
                    skip_res=False
                )
            )

        self.layers.append( 
            Conv2dZeros(
                filters=int(2 * nz),
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="same",
                name="mask_conv"
            )
        )

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        y_mask = inputs
        h_prime = y_mask
        for l in self.layers:
            # h_prime = tf.recompute_grad(l)(h_prime)
            h_prime = l(h_prime)
            # tf.add_to_collection("checkpoints", h_prime)

        return h_prime
