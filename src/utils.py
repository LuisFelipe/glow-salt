# -*- coding: utf-8 -*-
"""
module utils.py
--------------------
A set of utility functions to be used in the model implementations.
"""
import tensorflow as tf
import numpy as np
import os
from PIL import Image



def partial_restore(save_path):
    from tensorflow.python import pywrap_tensorflow
    from tensorflow.python.tools import inspect_checkpoint as chkp
    # chkp.print_tensors_in_checkpoint_file()
    reader = pywrap_tensorflow.NewCheckpointReader(save_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    all_vars = tf.global_variables()
    allvars_dic = dict()

    for var in all_vars:
        allvars_dic[var.name[:-2]] = var
    var_list = list()
    for key in sorted(var_to_shape_map):
        t = reader.get_tensor(key)
        var_list.append(tf.assign(allvars_dic[key], t))
    return tf.group(var_list)


def mkdir_if_not_exists(path):
    """
    make directory if it does not exists.
    :param path: dir path.
    :return: True if the path was created. False otherwise.
    """
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False


def save_figure_from_array(path, array):
    """
    Saves the array as image file.
    :param path: file path.
    :param array: figure numpy array.
    :return: True if success.
    """
    if array.shape[-1] == 1:
        # array = np.concatenate([array, array, array], axis=-1)
        array = np.squeeze(array)
        array = array.astype('uint8')
        img = Image.fromarray(array, mode='L')
    else:
        array = array.astype('uint8')
        img = Image.fromarray(array, mode='RGB')

    # img = img.convert(mode="L")
    img.save(fp=path)
    img.close()


def get_z_and_eps_size(config):
    """
    calculates the channel size for the latent variables z and eps.
    :param config: main program config object.
    :return: the size of z, eps and the latent resolution
    """
    # z_shape calculation
    h_resolution = int(config.get("dataset.h_resolution", 32))
    w_resolution = int(config.get("dataset.w_resolution", 32))
    channels = int(config.get("dataset.channels", 1))
    n_levels = int(config.get("model.levels", 6))
    space_reduction_factor = 2 ** n_levels
    latent_size_h = h_resolution // space_reduction_factor
    latent_size_w = w_resolution // space_reduction_factor
    z_channels = 2 * (channels * space_reduction_factor)
    eps_channels = channels * 2 ** (2 * n_levels)
    eps_channels -= z_channels
    return z_channels, eps_channels, (latent_size_h, latent_size_w)


def split_z_and_eps(latent, z_channels):
    _z = latent[:, :, :, :z_channels]
    _eps = latent[:, :, :, z_channels:]
    return _z, _eps


def manipulate(encoding, delta_z, z_size, weight=1.0, z_weight=1.0):
    # random choose perfil of noise
    # choice = np.random.choice(len(delta_z), size=1)
    # z_delta = delta_z[choice[0]]
    # z_delta = delta_z
    # delta_z, delta_eps = split_z_and_eps(z_delta, z_size)

    # delta_z, delta_eps = split_z_and_eps(np.expand_dims(z_delta, axis=0), z_size)
    # delta_z = np.squeeze(delta_z)
    # delta_eps = np.squeeze(delta_eps)

    # encoding_z, encoding_eps = split_z_and_eps(encoding, z_size)

    # manipulate
    # manipulated_z = (z_weight * encoding_z) + (weight * delta_z)
    # manipulated_eps = (z_weight * encoding_eps) + (weight * delta_eps)

    manipulated = (z_weight * encoding) + (weight * delta_z)
    manipulated_z, manipulated_eps = split_z_and_eps(manipulated, z_size)

    # split manipulated into z and eps
    # z_manipulated = manipulated_z[:, :, :, :z_channels]
    # eps_manipulated = manipulated_z[:, :, :, z_channels:]
    return manipulated_z, manipulated_eps
    # return manipulated_z, encoding_ep


def clear_callbacks():
    from flow.callbacks import on_epoch_begin, on_epoch_end, on_batch_begin, on_batch_end, on_train_begin, on_train_end, validate_sig, on_validate_begin, on_validate_end, before_session_initialization
    on_epoch_begin._clear_state()
    on_epoch_end._clear_state()
    on_batch_begin._clear_state()
    on_batch_end._clear_state()
    on_train_begin._clear_state()
    on_train_end._clear_state()
    validate_sig._clear_state()
    on_validate_begin._clear_state()
    on_validate_end._clear_state()
    before_session_initialization._clear_state()


def initialize_variables():
    """Default session initialization function."""
    # tf global variables initialization (session variables initialization)
    # sess = tf.get_default_session()
    # sess.run(tf.global_variables_initializer())
    # self.model._is_session_initialized = True
    sess = tf.get_default_session()
    not_initialized = sess.run([tf.is_variable_initialized(var) for var in tf.global_variables()])
    not_initialized = [v for (v, f) in zip(tf.global_variables(), not_initialized) if not f]
    if len(not_initialized) > 0:
        sess.run(tf.variables_initializer(not_initialized))


def get_distribution_strategy():
    if tf.distribute.in_cross_replica_context():
        strategy = tf.distribute.get_strategy()
    else:
        # to get strategy
        strategy = tf.distribute.get_replica_context()

    return strategy
