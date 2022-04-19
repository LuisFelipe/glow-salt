# -*- coding: utf-8 -*-
"""
module checkboard_spliter.py
-----------------------------
Z spliter using checkboard.
"""
import tensorflow as tf


class CheckerboardSplit(object):
    def forward(self, x, **kwargs):
        # assert isinstance(x, tf.Tensor)
        B, H, W, C = x.shape.as_list()
        x = tf.reshape(x, [-1, H, W // 2, 2, C])
        a, b = tf.unstack(x, axis=3)
        # assert a.shape == b.shape == [B, H, W // 2, C]
        return a, b

    def inverse(self, y, **kwargs):
        # assert isinstance(y, tuple)
        a, b = y
        # assert a.shape == b.shape
        B, H, W_half, C = a.shape.as_list()
        x = tf.stack([a, b], axis=3)
        # assert x.shape == [B, H, W_half, 2, C]
        return tf.reshape(x, [-1, H, W_half * 2, C])


class ChannelSplit(object):
    def forward(self, z, **kwargs):
        # assert isinstance(x, tf.Tensor)
        # split tensor along channel axis
        z_shape = z.shape.as_list()
        nz = z_shape[3]
        z1 = z[:, :, :, :nz // 2]
        z2 = z[:, :, :, nz // 2:]
        return z1, z2

    def inverse(self, z, **kwargs):
        # assert isinstance(y, tuple)
        z1, z2 = z

        return tf.concat([z1, z2], axis=3)