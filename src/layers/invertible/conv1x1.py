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
    The `ConvolutionPermute` normalizing flow Layer.
    """
    def __init__(self, name=None, det_type="LU", **kwargs):
        """
        Initialization function.
        :param name: layer name.
        """
        # Sample a random orthogonal matrix:
        super().__init__(name=name, )
        self.det_type = det_type

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
        
        if self.det_type == "LU":
            self.build_lu()
        else:
            self.build_cholesky()

        super().build(input_shape)
        self.built = True

    def build_lu(self):
        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(
            np.random.randn(*self.w_shape)
        )
        w_init = w_init[0].astype('float32')
        np_p, np_l, np_u = scipy.linalg.lu(w_init)
        self.l = self.add_weight(
            name=self.name + ".l",
            dtype=tf.float32,
            shape=self.w_shape,
            initializer=lambda shape, dtype: np_l,
            trainable=True
        )

        self.u = self.add_weight(
            name=self.name + ".u",
            dtype=tf.float32,
            shape=self.w_shape,
            initializer=lambda shape, dtype: np.triu(np_u, k=1),
            trainable=True
        )

        self.p = self.add_weight(
            name=self.name + ".p",
            dtype=tf.float32,
            shape=self.w_shape,
            initializer=lambda shape, dtype: np_p,
            trainable=False
        )

        self.log_s = self.add_weight(
            name=self.name + ".s",
            dtype=tf.float32,
            shape=self.w_shape[0],
            initializer=lambda shape, dtype: np.log(abs(np.diag(np_u))),
            trainable=True
        )

        self.sign_s = self.add_weight(
            name=self.name + ".s",
            dtype=tf.float32,
            shape=self.w_shape[0],
            initializer=lambda shape, dtype: np.sign(np.diag(np_u)),
            trainable=False
        )
        
        self.mask = np.tril(
            np.ones(
                self.w_shape, 
                dtype=np.float32
            ), 
            -1
        )

    def build_cholesky(self):
        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*self.w_shape))
        w_init = w_init[0].astype('float32')
        self.w = self.add_weight(
            name=self.name + ".w",
            dtype=tf.float32,
            shape=self.w_shape,
            initializer=lambda shape, dtype: w_init,
            trainable=True
        )

    def _get_w_and_logdet(self, z_shape, backwards=False):
        l = self.l * self.mask + np.eye(*self.w_shape, dtype=np.float32)
        u = self.u * self.mask.T + tf.linalg.diag(self.sign_s * tf.exp(self.log_s))
        
        if not backwards:
            _w = tf.matmul(l, u)
            _w = tf.matmul(self.p, _w)
        else:
            _w = tf.matmul(l, u)
            _w = tf.matmul(self.p, _w)
            # inverse matrix calculation
            _w = tf.matrix_inverse(_w)
            # u = tf.matrix_inverse(u)
            # l = tf.matrix_inverse(l)
            # p = tf.matrix_inverse(self.p)
            # _w = tf.matmul(l, p)
            # _w = tf.matmul(u, _w)

        d_logdet = tf.reduce_sum(self.log_s) * z_shape[1] * z_shape[2]

        # d_logdet = tf.cast(
        #     tf.math.log(
        #         abs(tf.linalg.det(tf.cast(self.w, 'float64')))
        #     ), dtype=tf.float32
        # ) * z_shape[1] * z_shape[2]
        return _w, d_logdet

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

        z_shape = z.shape.as_list()
        _w, d_logdet = self._get_w_and_logdet(z_shape)

        _w = tf.reshape(_w, [1, 1] + self.w_shape)

        z = tf.nn.conv2d(
            input=z,
            filters=_w,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )
        logdet += d_logdet

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

        _w, d_logdet = self._get_w_and_logdet(z_shape, backwards=True)

        # _w = tf.matrix_inverse(_w)
        _w = tf.reshape(_w, [1, 1] + self.w_shape)
        z = tf.nn.conv2d(
            input=z,
            filters=_w,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )

        logdet -= d_logdet

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


# class MatvecLU(bijector.Bijector):
#   """Matrix-vector multiply using LU decomposition.
#   """
#   def __init__(
#         self,
#         lower_upper,
#         permutation,
#         validate_args=False,
#         name=None
#     ):
#     """Creates the MatvecLU bijector.
#     Args:
#       lower_upper: The LU factorization as returned by `tf.linalg.lu`.
#       permutation: The LU factorization permutation as returned by
#         `tf.linalg.lu`.
#       validate_args: Python `bool` indicating whether arguments should be
#         checked for correctness.
#         Default value: `False`.
#       name: Python `str` name given to ops managed by this object.
#         Default value: `None` (i.e., 'MatvecLU').
#     Raises:
#       ValueError: If both/neither `channels` and `lower_upper`/`permutation` are
#         specified.
#     """
#         self._lower_upper = tensor_util.convert_nonref_to_tensor(
#               lower_upper, dtype_hint=tf.float32, name='lower_upper')

#       self._permutation = tensor_util.convert_nonref_to_tensor(
#           permutation, dtype_hint=tf.int32, name='permutation')
      
#       super(MatvecLU, self).__init__(
#           dtype=self._lower_upper.dtype,
#           is_constant_jacobian=True,
#           forward_min_event_ndims=1,
#           validate_args=validate_args,
#           name=name)

#   @property
#   def lower_upper(self):
#     return self._lower_upper

#   @property
#   def permutation(self):
#     return self._permutation

#   def _broadcast_params(self):
#     lower_upper = tf.convert_to_tensor(self.lower_upper)
#     perm = tf.convert_to_tensor(self.permutation)
#     shape = tf.broadcast_dynamic_shape(tf.shape(lower_upper)[:-1],tf.shape(perm))
#     lower_upper = tf.broadcast_to(lower_upper, tf.concat([shape, shape[-1:]], 0))
#     perm = tf.broadcast_to(perm, shape)
#     return lower_upper, perm

#   def _forward(self, x):
#     lu, perm = self._broadcast_params()
#     w = lu_reconstruct(lower_upper=lu,
#                        perm=perm,
#                        validate_args=self.validate_args)
#     return tf.linalg.matvec(w, x)

#   def _inverse(self, y):
#     lu, perm = self._broadcast_params()
#     return lu_solve(
#         lower_upper=lu,
#         perm=perm,
#         rhs=y[..., tf.newaxis],
#         validate_args=self.validate_args)[..., 0]

#   def _forward_log_det_jacobian(self, unused_x):
#     return tf.reduce_sum(
#         tf.math.log(tf.abs(tf.linalg.diag_part(self.lower_upper))),
#         axis=-1)

#   def _parameter_control_dependencies(self, is_init):
#     if not self.validate_args:
#       return []

#     lu, perm = None, None
#     assertions = []
#     if (is_init != tensor_util.is_ref(self.lower_upper) or
#         is_init != tensor_util.is_ref(self.permutation)):
#       lu, perm = self._broadcast_params()
#       assertions.extend(lu_reconstruct_assertions(
#           lu, perm, self.validate_args))

#     if is_init != tensor_util.is_ref(self.lower_upper):
#       lu = tf.convert_to_tensor(self.lower_upper) if lu is None else lu
#       assertions.append(assert_util.assert_none_equal(
#           tf.linalg.diag_part(lu), tf.zeros([], dtype=lu.dtype),
#           message='Invertible `lower_upper` must have nonzero diagonal.'))

#     return assertions

# #### Examples
#   #Here's an example of initialization via random weights matrix:
#   def trainable_lu_factorization(
#       event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
#     with tf.name_scope(name or 'trainable_lu_factorization'):
#       event_size = tf.convert_to_tensor(
#           event_size, dtype_hint=tf.int32, name='event_size')
#       batch_shape = tf.convert_to_tensor(
#           batch_shape, dtype_hint=event_size.dtype, name='batch_shape')
#       random_matrix = tf.random.uniform(
#           shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
#           dtype=dtype,
#           seed=seed)
#       random_orthonormal = tf.linalg.qr(random_matrix)[0]
#       lower_upper, permutation = tf.linalg.lu(random_orthonormal)
#       lower_upper = tf.Variable(
#           initial_value=lower_upper,
#           trainable=True,
#           name='lower_upper')
#       # Initialize a non-trainable variable for the permutation indices so
#       # that its value isn't re-sampled from run-to-run.
#       permutation = tf.Variable(
#           initial_value=permutation,
#           trainable=False,
#           name='permutation')
#       return lower_upper, permutation
#   channels = 3
#   conv1x1 = tfb.MatvecLU(*trainable_lu_factorization(channels),
#                          validate_args=True)
#   x = tf.random.uniform(shape=[2, 28, 28, channels])
#   fwd = conv1x1.forward(x)
#   rev_fwd = conv1x1.inverse(fwd)