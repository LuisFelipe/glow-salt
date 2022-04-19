# -*- coding: utf-8 -*-
"""
module sfgan.py
-----------------------------
Neural network blocks used inside the image_transformer model.
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
import numpy as np
from .layer_normalization import LayerNormalization
layers = tf.keras.layers


class SelfAttention(layers.Layer):
    """
    Image Transformer multihead Local Self-Attention Layer
    """

    def __init__(
            self, qkv_units, kernel_size, num_heads, dropout_rate=-1.0,
            normalize=True, out_units=None, skip_res=False, gatted=False,
            name=None, **kwargs
    ):
        self.kernel_size = kernel_size
        self.qkv_units = qkv_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.normalize = normalize
        self.out_units = out_units
        self.skip_res = skip_res
        self.gated = gatted
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return

        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) == 3:
                x_shape, orig, mask_shape = input_shape
            else:
                x_shape, orig = input_shape
        else:
            x_shape = input_shape
            orig = x_shape

        self.kqv_conv = layers.Conv2D(
            filters=3 * self.qkv_units * self.num_heads,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation=None
        )

        if self.out_units is not None:
            if self.gated:
                self.attention_deconv = layers.Conv2D(
                    filters=self.out_units * 2,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    use_bias=False,
                    activation=None
                )
            else:
                self.attention_deconv = layers.Conv2D(
                    filters=self.out_units,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    use_bias=False,
                    activation=None
                )
        else:
            if self.gated:
                self.attention_deconv = layers.Conv2D(
                    filters=orig.as_list()[-1] * 2,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    use_bias=False,
                    activation=None
                )
            else:
                self.attention_deconv = layers.Conv2D(
                    filters=orig.as_list()[-1],
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    use_bias=False,
                    activation=None
                )

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate, name=self.name + ".att_dropout")
            self.out_dropout = layers.Dropout(self.dropout_rate, name=self.name + ".dropout")

        if self.normalize:
            self.layer_norm = LayerNormalization()

        super().build(input_shape)
        self.built = True

    def call(self, inputs, training=False):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                x, orig = inputs
                mask = None
            else:
                x, orig, mask = inputs
                x_shape = x.shape.as_list()
                mask = tf.reshape(mask, shape=[x_shape[0], 1, 1, -1])
        else:
            x = inputs
            orig = x
            mask = None

        qkv = self.kqv_conv(x)
        # reshaping qkv to match the flowing shape:
        # [batch, num_heads, HW/4, channels / num_heads]
        qkv_headed = self.reshape_and_split_heads(qkv)
        q, k, v = tf.split(qkv_headed, 3, axis=-1)
        # q_headed = self.reshape_and_split_heads(q)
        # k_headed = self.reshape_and_split_heads(k)
        # v_headed = self.reshape_and_split_heads(v)
        # attention bias calculation

        ########################
        # attention calculation
        ########################
        attn_output = self.dot_product_attention([q, k, v], masks=mask)
        # output_shape = [n_batch, n_heads, w * h, v_channels]

        x_shape = x.shape.as_list()
        attn_output = self.reshape_and_concat_heads(attn_output, x_shape)
        # last output transform op
        # mix heads and put it back to the original shape
        if self.dropout_rate > 0.:
            attn_output = self.out_dropout(attn_output)
        attn_deconv = self.attention_deconv(attn_output)

        # attention gate
        if self.gated:
            a, b = tf.split(attn_deconv, 2, axis=-1)
            attn_deconv = a * tf.sigmoid(b)

        # self-attention skip connection
        if not self.skip_res:
            # otherwise, just adds the layer's input to the attended output
            output = attn_deconv + orig
        else:
            output = attn_deconv
        # layer normalization
        if self.normalize:
            output = self.layer_norm(output)

        return output

    def reshape_and_concat_heads(self, t, x_shape):
        t = tf.transpose(t, perm=[0,2,1,3])
        t = tf.reshape(
            t,
            shape=[
                x_shape[0],
                x_shape[1],
                x_shape[2],
                self.qkv_units * self.num_heads
            ]
        )
        return t

    def reshape_and_split_heads(self, x):
        """Split channels (dimension 3) into multiple heads (becomes dimension 1).

        :param x: a Tensor with shape [batch, height, width, channels]
        :param num_heads: an integer

        :return: a Tensor with shape [batch, num_heads, height* width, channels / num_heads]
        """
        # puts x in shape = [batch, num_heads, height, width, channels / num_heads]
        x_shape = x.shape.as_list()
        out = tf.reshape(
            x,
            shape=[
                x_shape[0],
                int(x_shape[1] * x_shape[2]),
                self.num_heads,
                x_shape[3] // self.num_heads
            ]
        )
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        return out

    def dot_product_attention(self, qkv, masks=None):
        """Dot-product attention.

        :param q: Tensor with shape [..., length_q, depth_k].
        :param k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
          match with q.
        :param v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
          match with q.
        :param bias: bias Tensor.
        bias is used to remove attention from padded units

        :return: Tensor with shape [..., length_q, depth_v].
        """
        [q, k, v] = qkv
        # attention q dot k
        logits = tf.matmul(q, k, transpose_b=True)
        # logits_shape = logits.shape.as_list()
        # logits /= np.sqrt(logits_shape[-1])
        logits /= np.sqrt(self.qkv_units)

        if masks is not None:
            logits -= (1 - masks) * 1e25

        # atteintion weights
        weights = tf.nn.softmax(logits, name=self.name + "_attention_weights", axis=-1)
        # weights = tf.nn.sigmoid(logits, name=self.name + "_attention_weights")

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout_rate > 0:
            weights = self.dropout(weights)

        return tf.matmul(weights, v)


class SelfAttention2(layers.Layer):
    """
    Image Transformer multihead Local Self-Attention Layer
    """

    def __init__(
            self, qkv_units, kernel_size, num_heads, dropout_rate=-1.0,
            name=None, **kwargs
    ):
        self.kernel_size = kernel_size
        self.qkv_units = qkv_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return

        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) == 3:
                x_shape, orig, mask_shape = input_shape
            else:
                x_shape, orig = input_shape
        else:
            x_shape = input_shape
            orig = x_shape

        self.kqv_conv = layers.Conv2D(
            filters=3 * self.qkv_units * self.num_heads,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation=None
        )

        self.attention_deconv = layers.Conv2D(
            filters=orig.as_list()[-1],
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            activation=None
        )

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate, name=self.name + ".att_dropout")

        super().build(input_shape)
        self.built = True

    def call(self, inputs, training=False):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                x, orig = inputs
                mask = None
            else:
                x, orig, mask = inputs
                x_shape = x.shape.as_list()
                mask = tf.reshape(mask, shape=[x_shape[0], 1, 1, -1])
        else:
            x = inputs
            orig = x
            mask = None

        qkv = self.kqv_conv(x)
        # reshaping qkv to match the flowing shape:
        # [batch, num_heads, HW/4, channels / num_heads]
        qkv_headed = self.reshape_and_split_heads(qkv)
        q, k, v = tf.split(qkv_headed, 3, axis=-1)
        # attention bias calculation

        ########################
        # attention calculation
        ########################
        attn_output = self.dot_product_attention([q, k, v], masks=mask)
        # output_shape = [n_batch, n_heads, w * h, v_channels]

        x_shape = x.shape.as_list()
        attn_output = self.reshape_and_concat_heads(attn_output, x_shape)
        # last output transform op
        # mix heads and put it back to the original shape
        attn_deconv = self.attention_deconv(attn_output)

        # self-attention skip connection
        output = attn_deconv + orig

        return output

    def reshape_and_concat_heads(self, t, x_shape):
        t = tf.transpose(t, perm=[0, 2, 1, 3])
        t = tf.reshape(
            t,
            shape=[
                x_shape[0],
                x_shape[1],
                x_shape[2],
                self.qkv_units * self.num_heads
            ]
        )
        return t

    def reshape_and_split_heads(self, x):
        """Split channels (dimension 3) into multiple heads (becomes dimension 1).

        :param x: a Tensor with shape [batch, height, width, channels]
        :param num_heads: an integer

        :return: a Tensor with shape [batch, num_heads, height* width, channels / num_heads]
        """
        # puts x in shape = [batch, num_heads, height, width, channels / num_heads]
        x_shape = x.shape.as_list()
        out = tf.reshape(
            x,
            shape=[
                x_shape[0],
                int(x_shape[1] * x_shape[2]),
                self.num_heads,
                x_shape[3] // self.num_heads
            ]
        )
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        return out

    def dot_product_attention(self, qkv, masks=None):
        """Dot-product attention.

        :param q: Tensor with shape [..., length_q, depth_k].
        :param k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
          match with q.
        :param v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
          match with q.
        :param bias: bias Tensor.
        bias is used to remove attention from padded units

        :return: Tensor with shape [..., length_q, depth_v].
        """
        [q, k, v] = qkv
        # attention q dot k
        logits = tf.matmul(q, k, transpose_b=True)
        # logits_shape = logits.shape.as_list()
        # logits /= np.sqrt(logits_shape[-1])
        logits /= np.sqrt(self.qkv_units)

        if masks is not None:
            logits -= (1 - masks) * 1e25

        # atteintion weights
        weights = tf.nn.softmax(logits, name=self.name + "_attention_weights", axis=-1)
        # weights = tf.nn.sigmoid(logits, name=self.name + "_attention_weights")

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout_rate > 0:
            weights = self.dropout(weights)

        return tf.matmul(weights, v)
