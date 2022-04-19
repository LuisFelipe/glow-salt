# -*- coding: utf-8 -*-
"""
module affine_nets.py
-----------------------------
Defines the Neural Network Layers used in the `S` and `T` operators 
of the Affine Layer. For more details we refer to the 
`Density estimation using Real NVP <https://arxiv.org/abs/1605.08803>`_ original article.
"""
import tensorflow as tf
from layers.layer_normalization import LayerNormalization
from layers.zero_init.conv2d_zeros import Conv2dZeros
layers = tf.keras.layers


class Block(layers.Layer):

    def __init__(self, filters, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters

    def build(self, input_shape):
        if self.built:
            return

        x_shape = input_shape.as_list()
        self.conv1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='same',
            activation=None,
        )   

        self.norm1 = LayerNormalization()

        self.conv2 = layers.Conv2D(
            filters=self.filters,
            kernel_size=[1, 1],
            padding='same',
            activation=None,
            strides=(1, 1)
        )

        self.norm2 = LayerNormalization()

        self.conv_zeros = Conv2dZeros(
            filters=x_shape[-1] * 2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            logscale_factor=3,
            skip=1
        )
        # self.layer_norm3 = LayerNormalization()
        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs

        x = self.conv1(x)
        x = self.norm1(x)

        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = tf.nn.relu(x)

        x = self.conv_zeros(x)

        return x


class Block2(layers.Layer):

    def __init__(self, filters, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.blocks = list()

    def build(self, input_shape):
        if self.built:
            return
        from layers.self_attention import SelfAttention

        layers = tf.keras.layers

        x_shape = input_shape.as_list()
        pos_shape = [x_shape[1], x_shape[2], x_shape[3]]
        # positional embedding
        self.pos_emb = self.add_weight(
            name=self.name + ".positional_emb",
            shape=pos_shape,
            dtype=tf.float32,
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            trainable=True
        )
        self.blocks.append(
            # layers.Lambda(function=lambda x: x + self.pos_emb)
            lambda x: x + self.pos_emb
        )

        # self-attention block
        self.blocks.append(
            SelfAttention(
                qkv_units=self.filters,
                kernel_size=(1, 1),
                num_heads=4,
                dropout_rate=0.1,
                normalize=False,
                # normalize=True,
                # skip_res=True,
                skip_res=False,
                gatted=False,
                # out_units=self.filters
            )
        )

        self.blocks.append(LayerNormalization(name=self.name + '.layer_norm1'))

        self.blocks.append(
            layers.Conv2D(
                filters=self.filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                use_bias=True,
                name=self.name + ".conv1"
            )
        )

        x_shape = input_shape.as_list()
        self.conv_zeros = Conv2dZeros(
            filters=x_shape[-1] * 2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            logscale_factor=3,
            skip=1,
        )

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs
        skip = x
        for l in self.blocks:
            # add residual connection before normalization
            if isinstance(l, LayerNormalization):
                x = x + skip

            # save skip connection
            skip = x
            # call layer
            x = l(x)

        x = self.conv_zeros(x)
        return x


class Block3(layers.Layer):

    def __init__(self, filters, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.blocks = list()

    def build(self, input_shape):
        if self.built:
            return
        from layers.self_attention import SelfAttention2

        layers = tf.keras.layers

        x_shape = input_shape.as_list()
        # self-attention block
        self.blocks.append(
            SelfAttention2(
                qkv_units=self.filters,
                kernel_size=(1, 1),
                num_heads=4,
                dropout_rate=0.1,
            )
        )

        self.blocks.append(LayerNormalization(name=self.name + '.layer_norm1'))

        self.blocks.append(
            layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                name=self.name + ".conv1"
            )
        )

        x_shape = input_shape.as_list()
        self.conv_zeros = Conv2dZeros(
            filters=x_shape[-1] * 2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            logscale_factor=3,
            skip=1
        )

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs

        for l in self.blocks:
            # call layer
            x = l(x)

        x = self.conv_zeros(x)
        return x
