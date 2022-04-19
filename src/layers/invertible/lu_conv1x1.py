# -*- coding: utf-8 -*-
"""
module conv1x1.py
-----------------------------
Invertible 1x1 Conv Layer as defined in:
`Glow: Better...`
"""
import tensorflow as tf
import numpy as np
import scipy
layers = tf.keras.layers


class Conv1x1(layers.Layer):
    """
    LU Decomposition version of the Invertible 1x1 Convolutional layer.
    """
    def __init__(self, name=None, **kwargs):
        """
        Initialization function.
        :param name: layer name.
        """
        # Sample a random orthogonal matrix:
        super().__init__(name=name, )

    def _lu_initialization(self):
        random_matrix = tf.random.uniform(
            shape=self.w_shape,
            dtype=tf.float32,
        )
        # orthonormal random matrix
        random_orthonormal = tf.linalg.qr(random_matrix)[0]
        # lu decomposition 
        lower_upper, permutation = tf.linalg.lu(random_orthonormal)
        # trainable lower_upper lu decomposable matrix
        self.lower_upper = tf.Variable(
          initial_value=lower_upper,
          trainable=True,
          name=self.name + '.lower_upper'
        )
        # Initialize a non-trainable variable for the permutation indices so
        # that its value isn't re-sampled from run-to-run.
        self.permutation = tf.Variable(
          initial_value=permutation,
          trainable=False,
          name='permutation'
        )

    def build(self, input_shape):
        if self.built:
            return

        if len(input_shape) == 3:
            z_shape, logdet_shape, _eps_shape = input_shape
        else:
            z_shape, logdet_shape = input_shape

        z_shape = z_shape.as_list()
        w_shape = [z_shape[3], z_shape[3]]
        self.w_shape = w_shape
        
        self._lu_initialization()
        super().build(input_shape)
        self.built = True

    def lu_reconstruct(self, lower_upper, perm):
        """The inverse LU decomposition, `X == lu_reconstruct(*tf.linalg.lu(X))`.

        :param lower_upper: `lu` as returned by `tf.linalg.lu`, i.e., if
              `matmul(P, matmul(L, U)) = X` then `lower_upper = L + U - eye`.
        :param perm: `p` as returned by `tf.linag.lu`, i.e., if
              `matmul(P, matmul(L, U)) = X` then `perm = argmax(P)`.
        :param validate_args: Python `bool` indicating whether arguments 
                              should be checked for correctness.
                              Default value: `False` (i.e., don't validate arguments).
        :return:
            x: The original input to `tf.linalg.lu`, i.e., `x` as in,
              `lu_reconstruct(*tf.linalg.lu(x))`.
        """
        lower_upper = tf.convert_to_tensor(
            lower_upper, 
            dtype_hint=tf.float32, 
            name='lower_upper'
        )
        perm = tf.convert_to_tensor(
            perm, 
            dtype_hint=tf.int32, 
            name='perm'
        )

        shape = tf.shape(lower_upper)

        lower = tf.linalg.set_diag(
            tf.linalg.band_part(
                lower_upper, 
                num_lower=-1, 
                num_upper=0
            ),
            tf.ones(
                shape[:-1], 
                dtype=lower_upper.dtype
            )
        )
        upper = tf.linalg.band_part(
            lower_upper, 
            num_lower=0, 
            num_upper=-1
        )

        _w = tf.matmul(lower, upper)
        _w = tf.gather(
            _w, 
            tf.math.invert_permutation(perm)
        )
        return _w

    def _log_det_jacobian(self):
        logdet = - tf.reduce_sum( 
            tf.math.log(
                tf.abs(
                    tf.linalg.diag_part(self.lower_upper)
                )
            ),
        ) * self.w_shape[0] * self.w_shape[1]
        return logdet

    def forward(self, inputs):
        """
        Forward pass of the reversible layer.
        :param inputs: input tensors
        :return: list of outputs
        """
        if len(inputs) == 3:
            z, logdet, _eps = inputs
        else:
            z, logdet = inputs
            _eps = None

        _w = self.lu_reconstruct(self.lower_upper, self.permutation)

        _w = tf.reshape(_w, [1, 1] + self.w_shape)

        z = tf.nn.conv2d(
            input=z,
            filters=_w,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )

        logdet += self._log_det_jacobian()

        if _eps is not None:
            return [z, logdet, _eps]
        else:
            return [z, logdet]

    def backwards(self, inputs):
        """
        Forward pass of the reversible layer.
        :param inputs: input tensors
        :return: list of outputs
        """
        if len(inputs) == 3:
            z, logdet, _eps = inputs
        else:
            z, logdet = inputs
            _eps = None

        z_shape = z.shape.as_list()

        _w = self.lu_reconstruct(self.lower_upper, self.permutation)
        logdet_j = self._log_det_jacobian()
        _w = tf.matrix_inverse(_w)
        _w = tf.reshape(_w, [1, 1] + self.w_shape)
        z = tf.nn.conv2d(
            input=z,
            filters=_w,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )

        logdet -= logdet_j

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
