# -*- coding: utf-8 -*-
"""
module salt_ds.py
--------------------
salt dataset reader and iterator.
"""
import os
import sys
import random
import warnings
import tensorflow as tf
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d 
from sklearn.model_selection import StratifiedShuffleSplit


class RawDataReader(object):
    """ Salt deposits and masks dataset. 
    Reads the dataset and iterates over it.
    """

    def __init__(
        self, config=None, 
        model_path=None, velocity_path=None, model_size=(1040, 7760),
        strip=None,striph=None, mask_threshold=None, patch_size=(64,64),
        partition="train", partition_bound=3850,
        bounded=True, *args, **kwargs
    ):
        """Dataset generator initialization."""
        self._idx = None
        self._x = list()
        self._y = list()
        self.partition = partition
        if len(partition_bound) == 1:
            partition_bound.append(-1)
        self._partition_bound = partition_bound
        self.model_path = model_path
        self.velocity_path = velocity_path
        self.model_size = model_size
        self.strip = strip
        self.striph = striph
        self.mask_threshold = mask_threshold
        self.patch_size = patch_size
        self.step_size = int(config.get("dataset.step_size", self.patch_size[0]))
        self.config = config
        self.random_state = np.random.RandomState(123456)
        self.cmap = plt.get_cmap("seismic")
        # self.cmap = plt.get_cmap("prism_r")
        self._raw_model = None
        self._raw_velocity = None
        self._mask = None
        self.max_size = int(self.config.get("dataset.max_size", -1))
        self.bounded = bounded
        # load img paths
        self.load_data()

    def _read_salt_model(self):
        """ Reads a salt deposit model from a binary file.
        """
        self._raw_model = self._read(self.model_path)
        # mean = np.mean(self._raw_model)
        # std = np.std(self._raw_model)
        # self._raw_model -= mean
        self._raw_model -= np.min(self._raw_model)
        # self._raw_model /= std
        self._raw_model /= np.max(self._raw_model)
        # norm = colors.Normalize(vmin=np.min(self._raw_model), vmax=np.max(self._raw_model))
        
        # norm = colors.DivergingNorm(vcenter=0.0)
        # self._raw_model = norm(self._raw_model)
        self._raw_model = self.cmap(self._raw_model)[:, :, :3]
        self._raw_velocity = self._read(self.velocity_path)

        # recovering masks
        self._mask = self._raw_velocity == self.mask_threshold
        self._mask = self._mask.astype(np.int)

    def _read(self, path):
        """ Reads a binary model file and returns.
        
        Reads a binary salt model file into a numpy tensor.
        :param path: path to the binary file.
        :type path: str
        :returns: a numpy matrix. 
        :rtype: numpy.ndArray
        """
        with open(path, "rb") as f:
            data = np.fromfile(f, dtype=np.float32)
            data = np.reshape(data, self.model_size)
            data = data.T
            if self.strip is not None:
                data = data[self.striph[0]:self.striph[1], self.strip[0]:self.strip[1]]
        return data

    def get_patches(self):
        """ Split a the salt model into patches.
        """
        self._x, self._y = np.meshgrid(
            np.arange(0, self._raw_model.shape[0], self.step_size),
            np.arange(0, self._raw_model.shape[1], self.step_size)
        )
        self._x = self._x.flatten()
        self._y = self._y.flatten()

    def _split_partition(self):
        """ Split data into Train, Test and validation sets. 
        
        Splits (actually crops) the salt raw data according to the curent dataset partition. 
        """
        if self.bounded:
            if self.partition == "train":
                self._raw_model = self._raw_model[:, :self._partition_bound[0]]
                self._raw_velocity = self._raw_velocity[:, :self._partition_bound[0]]
                self._mask = self._mask[:, :self._partition_bound[0]]
            elif self.partition == "valid":
                self._raw_model = self._raw_model[:, self._partition_bound[0]:self._partition_bound[1]]
                self._raw_velocity = self._raw_velocity[:, self._partition_bound[0]:self._partition_bound[1]]
                self._mask = self._mask[:, self._partition_bound[0]:self._partition_bound[1]]
            elif self.partition == "test":
                self._raw_model = self._raw_model[:, self._partition_bound[0]:self._partition_bound[1]]
                self._raw_velocity = self._raw_velocity[:, self._partition_bound[0]:self._partition_bound[1]]
                self._mask = self._mask[:, self._partition_bound[0]:self._partition_bound[1]]

    def load_data(self):
        """Dataset load main function.
        
        Dataset load and data initializer main function. 
        It reads the salt model and builds the patches used as dataset.  
        """
        self._read_salt_model()
        self._split_partition()
        self.get_patches()

        # deterministic shuffle ds
        perm = self.random_state.permutation(len(self._x))
        self._x = self._x[perm]
        self._y = self._y[perm]

        # if dataset.max_size is set
        # then we truncate the dataset length to fit the max_size
        
        if self.max_size > 0:
            self._x = self._x[:self.max_size]
            self._y = self._y[:self.max_size]

    def __len__(self):
        return len(self._x)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._x[item], self._y[item]

        elif isinstance(item, slice):
            return self._x[item], self._y[item]
        else:
            raise TypeError(
                "Dataset indices must be integers or slices, not {}.".format(type(item))
            )

    def __iter__(self):
        """ Restart the generator from the begining and returns it as a iterable.
        
        :returns: The current generator (self) as a iterable.
        """
        self._idx = 0
        return self

    def __next__(self):
        """ Returns the next generator item.
        
        :returns: A single sample input data x and it target y.
        :rtype: tuple
        :raises: StopIteration
        """
        ds_len = len(self)
        # stop iteration condition
        if self._idx == ds_len:
            raise StopIteration()

        xy = self._get_patch()
        
        if self._idx == len(self):
            raise StopIteration()
        
        x, y = xy
        # norm = colors.Normalize(vmin=np.min(x), vmax=np.max(x))
        # x = norm(x)
        # x = self.cmap(x)[:, :, :3]
        y = np.expand_dims(y, axis=-1)
        # go to next idx + 1
        self._idx += 1

        return x, y

    def _get_patch(self):
        while True:
            idx_x = self._x[self._idx]
            idx_y = self._y[self._idx]

            if (idx_x + self.patch_size[0]) < self._raw_model.shape[0] and (idx_y + self.patch_size[1]) < self._raw_model.shape[1]:
                break
            else:
                self._idx += 1
                if self._idx == len(self):
                    return

        x = self._raw_model[idx_x:idx_x+self.patch_size[0], idx_y:idx_y+self.patch_size[1]]
        y = self._mask[idx_x:idx_x+self.patch_size[0], idx_y:idx_y+self.patch_size[1]]

        if x.shape[0] != self.patch_size[0] or x.shape[1] != self.patch_size[1]:
            aux = np.zeros(shape=self.patch_size+[3], dtype=np.float32)
            aux[:x.shape[0], :x.shape[1]] = x[:, :]
            x = aux.copy()
            aux = np.zeros(shape=self.patch_size, dtype=np.float32)
            aux[:y.shape[0], :y.shape[1]] = y[:, :]
            y = aux.copy()
            # self._idx += 1

        return x, y

