# -*- coding: utf-8 -*-
"""
module factor_out.py
-----------------------------
The layer that factors out half of the variables
    directly to the latent space.
"""
import tensorflow as tf
import numpy as np
from layers.zero_init.conv2d_zeros import Conv2dZeros
from layers.self_attention import SelfAttention
layers = tf.keras.layers


class FactorOut(layers.Layer):
    """
    The layer that factors out half of the variables
    directly to the latent space. This layer can be used to model the
    multi-scale architecture.
    Important notes:
        * if z is not None it will be concatenated with factored out tensor
        * this is volume preserving operation, hence logdet is unchanged
    """

    def __init__(self, name: str = "", factor=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = factor
        self.prior_net = PriorNet(
            apply_selfattention=True,
            name=self.name + ".factor_priornet"
        )

    def _get_mean_and_logstd(self, z1):
        h = z1
        h = self.prior_net(h)

        mean = h[:, :, :, 0::2]
        logs = h[:, :, :, 1::2]

        return mean, logs

    def _compute_logp(self, z2, mean, logstd):
        log_ps = -0.5 * (
                np.log(2 * np.pi) + 2. * logstd + (z2 - mean) ** 2 / tf.exp(2. * logstd)
        )
        logp = tf.reduce_sum(log_ps, [1, 2, 3])
        return logp

    def forward(self, inputs, **kwargs):
        if len(inputs) == 3:
            z, logdet, _eps = inputs
        else:
            z, logdet = inputs
            _eps = None

        z_shape = z.shape.as_list()
        nz = z_shape[3]
        # saving split factor
        self.split_factor = nz // self.factor

        z1 = z[:, :, :, :self.split_factor]
        z2 = z[:, :, :, self.split_factor:]
        mean, logstd = self._get_mean_and_logstd(z1)
        logdet += self._compute_logp(z2, mean, logstd)
        new_eps = (z2 - mean) / tf.exp(logstd)

        if _eps is not None:
            _eps = tf.concat([new_eps, _eps], axis=3)
        else:
            _eps = new_eps
        return [z1, logdet, _eps]

    def backward(self, inputs, **kwargs):
        if len(inputs) == 3:
            z, logdet, _eps = inputs
        else:
            z, logdet = inputs
            _eps = None
            raise Exception("expected z, logdet, and _eps as input.")

        mean, logstd = self._get_mean_and_logstd(z)
        self.split_factor = z.shape[-1]
        eps1 = _eps[:, :, :, :self.split_factor]
        eps2 = _eps[:, :, :, self.split_factor:]

        z2 = mean + tf.exp(logstd) * eps1
        z = tf.concat([z, z2], 3)

        return [z, logdet, eps2]

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
            out = self.forward(inputs)
        else:
            out = self.backward(inputs)

        return out


class PriorNet(layers.Layer):
    """ Prior network layer.
    """
    def __init__(self, apply_selfattention=True, name=None, **kwargs):
        """
        Prior network initialization.
        :param name: layers unique identifier name.
        :param kwargs: kwargs dict like.
        """
        super().__init__(name=name, **kwargs)
        self._apply_selfattention = apply_selfattention

    def build(self, input_shape):
        if self.built:
            return
        z_shape = input_shape.as_list()
        self.layers = list()

        if self._apply_selfattention:
            self.pos_emb = self.add_weight(
                name="positional_emb",
                shape=[z_shape[1], z_shape[2], z_shape[3]],
                dtype=tf.float32,
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                trainable=True
            )
            self.layers.append(
                # tf.keras.layers.Lambda(function=lambda x: x + self.pos_emb)
                lambda x: x + self.pos_emb
            )

            self.layers.append(
                SelfAttention(
                    qkv_units=z_shape[3],
                    kernel_size=(1, 1),
                    num_heads=4,
                    dropout_rate=0.1,
                    normalize=True,
                    gatted=False,
                    # out_units=z_shape[3] * 2,
                    skip_res=False
                )
            )

            self.layers.append(
                tf.keras.layers.Conv2D(
                    filters=z_shape[3],
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    use_bias=True,
                    activation=tf.nn.relu
                )
            )

        self.layers.append(
            Conv2dZeros(
                filters=z_shape[-1] * 2,
                kernel_size=[3, 3],
                strides=(1, 1),
                padding="SAME",
                logscale_factor=3,
                skip=1,
                name=self.name + ".conv_zeros"
            )
        )

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        z1 = inputs
        h = z1
        for l in self.layers:
            h = l(h)
        # tf.add_to_collection("checkpoints", h)

        return h
