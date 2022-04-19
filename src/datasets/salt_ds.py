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
from sklearn.model_selection import StratifiedShuffleSplit


class SaltDS(object):
    """ Salt deposits and masks dataset. 
    Reads the dataset and iterates over it.
    """

    def __init__(
        self, config=None, 
        path=None, partition="train",
        *args, **kwargs
    ):
        """Main module initialization."""
        self._idx = None
        self._x = list()
        self._y = list()
        self.partition = partition
        self.path = path
        self.config = config
        self.random_state = np.random.RandomState(123456)
        self.mask_channels = int(self.config.get("dataset.mask_channels", 2))
        self.cmap = plt.get_cmap("seismic")
        # load img paths
        self.load_data()
        if self.partition == "all":
            self.partition = "train"
        self.dataset = None

    def load_data(self):
        max_size = int(self.config.get("dataset.max_size", -1))
        valid_size = float(self.config.get("dataset.valid_size", 0.1))
        should_augment = self.config.get("dataset.augment", "false").strip().lower() == "true" 

        for root, _, files in os.walk(self.path):
            for file in files:
                if not file.endswith(".npy") or file.endswith("_mask.npy"):
                    continue
                p = os.path.join(root, file)
                self._x.append(np.load(p))
                p = os.path.join(root, file.replace(".npy", "_mask.npy"))
                self._y.append(np.load(p))

        # as np array
        self._y = np.concatenate(self._y, axis=0)
        self._x = np.concatenate(self._x, axis=0)

        # deterministic shuffle ds
        perm = self.random_state.permutation(len(self._x))
        self._x = self._x[perm]
        self._y = self._y[perm]

        self._meta = np.stack(
            [
                np.arange(0, len(self._x), dtype=np.int),
                np.zeros(shape=(len(self._x)), dtype=np.int)
            ],
            axis=-1
        )

        if self.partition != "test" and valid_size > 0:
            self._split_train_test(valid_size)

        # if dataset.max_size is set
        # then we truncate the dataset length to fit the max_size
        if max_size > 0:
            self._x = self._x[:max_size]
            self._y = self._y[:max_size]
            self._meta = self._meta[:max_size]

        if should_augment:
            self._meta = np.tile(self._meta, (4, 1))
            self._meta[len(self._x):, 1] = 1
            self._meta[2 * len(self._x):, 1] = 2
            self._meta[3 * len(self._x):, 1] = 3

    def _split_train_test(self, valid_size):
        """
        Splits the dataset into train-test partitions.
        :param valid_size: validation/test desired size.
        """
        # train test split
        spliter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=654321)
        [(train_idx, test_idx)] = list(spliter.split(self._x, self._y))
        if self.partition == "train":
            self._x = self._x[train_idx]
            self._y = self._y[train_idx]
        else:
            self._x = self._x[test_idx]
            self._y = self._y[test_idx]

    def __len__(self):
        return len(self._meta)

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
        self._idx = 0
        # random shuffle dataset on stop iteration
        perm = self.random_state.permutation(len(self))
        self._meta = self._meta[perm]
        return self

    def __next__(self):
        ds_len = len(self)
        # stop iteration condition
        if self._idx >= ds_len:
            raise StopIteration()

        _idx, augmentation_type = self._meta[self._idx]

        # x = (self._x[_idx] + 0.5) * 255
        x = self._x[_idx]
        y = self._y[_idx]

        x, y = self.augment(augmentation_type, x, y)
        x = np.squeeze(x, axis=2)

        norm = colors.Normalize(vmin=np.min(x), vmax=np.max(x))
        x = norm(x)
        x = self.cmap(x)[:, :, :3] * 255
        # if mask_channels is bigger than one, then it will have exactly two channes
        # composed by the mask and its complement.
        if self.mask_channels > 1:
            y = np.concatenate([y, 1 - y], axis=-1) 

        # if x channels is bigger than 1
        # if self._inputs_config["output_shapes"][0][-1] > 1:
        #     # then repeats x values across the channels.
        #     # this is a trick used to increase the number of learnable network parameters.  
        #     x = np.tile(
        #         x, 
        #         reps=[
        #             1,
        #             1, 
        #             self._inputs_config["output_shapes"][0][-1]
        #         ]
        #     )

        # go to next idx + 1
        self._idx += 1

        return x, y

    def build_dataset(self, inputs):
        # build output type list
        output_types = [x.dtype for x in inputs]
        output_shapes = [x.shape for x in inputs]
        # build 
        batch_size = int(self.config.get("FLOW.BATCH_SIZE", 1))
        prefetch_buffer = int(self.config.get("DATASET.BUFFER_SIZE", 1))

        dataset = tf.data.Dataset.from_generator(
            generator=lambda: iter(self),
            output_types=output_types,
            output_shapes=output_shapes
        )
        dataset = dataset.batch(
            batch_size.
            drop_remainder=True
        )
        self.dataset = dataset.prefetch(buffer_size=prefetch_buffer)
        return self.dataset

    def read_img(self, path):
        x = np.load(path)
        return x

    def augment(self, augmentation_type, x, y):
        if augmentation_type == 1:
            x = np.fliplr(x)
            y = np.fliplr(y)
        elif augmentation_type == 2:
            x = np.flipud(x)
            y = np.flipud(y)
        elif augmentation_type == 3:
            x = np.fliplr(x)
            y = np.fliplr(y)
            x = np.flipud(x)
            y = np.flipud(y)

        return x, y
