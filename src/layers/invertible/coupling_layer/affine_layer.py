# -*- coding: utf-8 -*-
"""
module affine_layer.py
-----------------------------
Affine Layer as defined on the article:
`DENSITY ESTIMATION USING REAL NVP`
"""
import tensorflow as tf
layers = tf.keras.layers


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


class CouplingLayer(layers.Layer):
    """
    A affine coupling layer. Reversible layer implementation.
    """

    def __init__(
            self,
            filters,
            level_num,
            block_num,
            name=None,
            block_type="block1",
            **kwargs
    ):
        """
        Implementation of the affine coupling layer (see RealNVP or NICE) paper
        for more details.
        Args:
            shift_and_log_scale_fn: a function which takes for the input tensor
                of shape [batch_size, width, height, num_channels] and return
                shift and log_scale tensors of the same shape.
            name: a custom name of the flow
            log_scale_fn: a log scale function
            **kwargs:
        """
        super().__init__(name=name, **kwargs)
        self._output_shape = None
        self.filters = filters
        self.level_num = level_num
        self.block_num = block_num
        self.block_type = block_type
        self._build_nn_block()
        self._f = self._f
        self.channel_split = ChannelSplit()

    def _build_nn_block(self):
        if self.block_type == "block2":
            from .affine_nets import Block2
            self._f = Block2(
                filters=self.filters
            )

        elif self.block_type == "block1":
            from .affine_nets import Block
            self._f = Block(
                filters=self.filters
            )
        elif self.block_type == "block3":
            from .affine_nets import Block3
            self._f = Block3(
                filters=self.filters
            )
        else:
            raise ValueError(
                "block_type not recognized: '{}' \n"
                "Known values: 'block1', 'block2', 'block3'.".format(self.block_type)
            )

    def transform(self, z):
        h = self._f(z)

        shift = h[:, :, :, 0::2]
        scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
        # scale = tf.nn.exp(h[:, :, :, 1::2])
        return shift, scale

    def forward(self, inputs):
        if len(inputs) == 3:
            # x, logdet, z = inputs
            z, logdet, _eps = inputs
        else:
            # x, logdet = inputs
            z, logdet = inputs
            _eps = None

        z_shape = z.shape.as_list()
        nz = z_shape[3]
        # split tensor along channel axis
        # z1 = z[:, :, :, :nz // 2]
        # z2 = z[:, :, :, nz // 2:]
        z1, z2 = self.channel_split.forward(z)

        # affine transformation
        shift, scale = self.transform(z1)
        z2 += shift
        z2 *= scale
        
        logdet += tf.reduce_sum(
            tf.math.log(scale), 
            axis=[1, 2, 3]
        )

        # z = tf.concat([z1, z2], axis=3)
        z = self.channel_split.inverse((z1, z2))
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

        z_shape = z.shape.as_list()
        nz = z_shape[3]
        # z1 = z[:, :, :, :nz // 2]
        # z2 = z[:, :, :, nz // 2:]
        z1, z2 = self.channel_split.forward(z)

        # Inverse Affine Transformation
        shift, scale = self.transform(z1)
        z2 /= scale
        z2 -= shift
        logdet -= tf.reduce_sum(
            tf.math.log(scale), 
            axis=[1, 2, 3]
        )

        # z = tf.concat([z1, z2], 3)
        z = self.channel_split.inverse((z1, z2))

        if _eps is not None:
            return [z, logdet, _eps]
        else:
            return [z, logdet]

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
            out = self.backwards(inputs)

        return out
