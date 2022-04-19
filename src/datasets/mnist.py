# -*- coding: utf-8 -*-
"""
module minist.py
--------------------
minist dataset reader.
"""
import os
import sys
import random
import warnings
import tensorflow as tf
import numpy as np
import pickle
import numpy as np
from flow.dataset import Dataset as DS


class MnistDS(DS):
    """ Project main class.
    """
    def __init__(self, inputs_config, config=None, partition="train", path=None, iterator=None, *args, **kwargs):
        """Main module initialization."""
        super().__init__(inputs_config, config, path, iterator, partition, *args, **kwargs)
        self.partition = partition
        self.repeat = False
        self.path = path
        self._idx = None
        self.load_data()

    def load_data(self):
        mnist = tf.keras.datasets.mnist
        path = os.getcwd().replace("src", "")
        (x_train, y_train), (x_test, y_test) = mnist.load_data(path + "data/mnist.npz")
        y_train = np.reshape(y_train, [-1])
        y_test = np.reshape(y_test, [-1])
        # Pad with zeros to make 32x32
        x_train = np.lib.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'minimum')
        # Pad with zeros to make 32x23
        x_test = np.lib.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'minimum')
        x_train = np.tile(np.reshape(x_train, (-1, 32, 32, 1)), (1, 1, 1, 3))
        # x_train = np.reshape(x_train, (-1, 32, 32, 1))
        x_test = np.tile(np.reshape(x_test, (-1, 32, 32, 1)), (1, 1, 1, 3))
        # x_test = np.reshape(x_test, (-1, 32, 32, 1))

        print('n_shard_train:', x_train.shape[0], 'n_shard_test:', x_test.shape[0])
        print('train_shape:', x_train.shape, 'test:', x_test.shape)
        if self.partition == "train":
            self._x = x_train
            # self._x_test = x_test
            self._y = y_train
            # self._y_test = y_test
        elif self.partition == "valid":
            self._x = x_test
            self._y = y_test
        else:
            raise ValueError("Partition name not recognized. partition={}.".format(self.partition))

        max_samples = int(self.config.get("dataset.max_size", -1))
        if max_samples > 0:
            self._x = self._x[:max_samples]
            # self._x_test = self._x_test[:max_samples]
            self._y = self._y[:max_samples]
            # self._y_test = self._y_test[:max_samples]

        print(">>>>>", self._x.shape)
        print(">>>>>", self._y.shape)

    def build_dataset(self):
        batch_size = int(self.config.get("FLOW.BATCH_SIZE", 1))
        prefetch_buffer = 40
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: iter(self),
            output_types=self._inputs_config["output_types"],
            output_shapes=self._inputs_config["output_shapes"]
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer)
        return dataset

    def __len__(self):
        return len(self._x)

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        ds_len = len(self._x)
        batch_size = int(self.config.get("FLOW.BATCH_SIZE", 1))
        mod_batch = ds_len % batch_size
        # stop iteration condition
        if self._idx >= (ds_len - mod_batch):
            print(">>>>>>>end_of_seq>>>>", len(self._x), self.partition, self._idx, batch_size, mod_batch, self.repeat)
            p = np.random.permutation(len(self._x))
            self._x_train = self._x[p]
            self._y_train = self._y[p]
            if self.repeat:
                self._idx = 0
            else:
                raise StopIteration()

        x, y = self._x[self._idx], self._y[self._idx]

        x = x.astype(np.float32)
        resolution = int(self.config.get("dataset.resolution", 32))
        x = downsample(x, resolution)
        x = x.astype(np.float32)
        self._idx += 1
        return x, y


def downsample(x, resolution):
    assert x.dtype == np.float32
    assert x.shape[0] % resolution == 0
    assert x.shape[1] % resolution == 0
    if x.shape[1] == x.shape[2] == resolution:
        return x
    s = x.shape
    x = np.reshape(x, [resolution, s[0] // resolution,
                       resolution, s[1] // resolution, s[2]])
    x = np.mean(x, (1, 3))
    return x


def x_to_uint8(x):
    x = np.clip(np.floor(x), 0, 255)
    return x.astype(np.uint8)


def shard(data, shards, rank):
    # Determinisitc shards
    x, y = data
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] % shards == 0
    assert 0 <= rank < shards
    size = x.shape[0] // shards
    ind = rank*size
    return x[ind:ind+size], y[ind:ind+size]





