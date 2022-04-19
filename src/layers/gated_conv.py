# -*- coding: utf-8 -*-
"""
module gated_conv.py
-----------------------------
Gated Convolutional block.
"""
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
layers = tf.keras.layers


class GatedConv(layers.Layer):
    """
    Gated Convolutional Block.
    """

    def __init__(
            self, kernel_size, filters, dropout_rate=-1.0, strides=(1,1), padding="same",
            name=None, **kwargs
    ):
        self.kernel_size = kernel_size
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.strides = strides
        self.padding = padding
        super().__init__(name=name, **kwargs)
        self._init_sublayers()

    def _init_sublayers(self):
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=True,
            activation=None
        )

        self.conv2 = layers.Conv2D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_bias=True,
            activation=None
        )

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate, name=self.name + ".att_dropout")

    def call(self, inputs, training=False):
        x = inputs
        x_in = tf.nn.elu(tf.concat([x, -x], axis=-1))
        out = self.conv(x_in)

        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.conv2(out)
        out_a, out_b = tf.split(out, 2, axis=-1)
        # gate multiply
        out = out_a * tf.nn.sigmoid(out_b)
        # residual connection
        out += x
        return out
