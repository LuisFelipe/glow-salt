# -*- coding: utf-8 -*-
"""
module salt_dsv2.py
--------------------
salt dataset reader and iterator second version.
It reads the dataset from a preprocessed data path.
Additionally, it can be used to augment the data with horizontal flips.
"""
import os
import tensorflow as tf
import numpy as np
from PIL import Image


class SaltDS(object):
    """Seismogram dataset iterator.
    Reads the dataset and iterates over it.
    """

    def __init__(self, config=None, path=None, partition="train", *args, **kwargs):
        """Main module initialization."""
        self._idx = None
        self._meta = None
        self._x = list()
        self._y = list()
        self._x_augs = list()
        self._y_augs = list()
        self.output_names = ("x", "y")

        self.partition = partition
        self.path = path
        self.config = config
        self.random_state = np.random.RandomState(1234567)

        self._init_configs()
        # load img paths
        self.load_data()
        if partition == "all":
            partition = "train"
        self.dataset = None
        self.build_dataset()

    def _init_configs(self):
        self.max_size = int(self.config.get("dataset.max_size", -1))
        self.flip_augment = self.config.get("dataset.augment", "false").strip().lower() == "true"
        self.batch_size = int(self.config.get("flow.global_batch_size", 1))
        self.prefetch_buffer = int(self.config.get("DATASET.BUFFER_SIZE", 1))
        self.patch_size = [int(self.config.get("dataset.h_resolution")), int(self.config.get("dataset.w_resolution"))]
        self.valid_size = float(self.config.get("dataset.valid_size", 0.1))
        self.mask_channels = int(self.config.get("dataset.mask_channels", 2))

    def load_data(self):
        if self.partition != "all":
            path = os.path.join(self.path, self.partition)
        else:
            path = self.path
        # load partition
        for root, dirs, files in os.walk(path):
            for f in files:
                if not f.endswith(".png"):
                    continue
                if "_mask" in f:
                    continue
                f_path = os.path.join(root, f)
                self._x.append(f_path)
                f_path = os.path.join(root, f.replace(".png", "_mask.png"))
                self._y.append(f_path)

        self._y = np.asarray(self._y)
        self._x = np.asarray(self._x)

        # flip augmentation config
        self._meta = np.zeros_like(self._x, dtype=np.int8)

        # if dataset.max_size is set
        # then we truncate the dataset length to fit the max_size
        if self.max_size > 0:
            self._x = self._x[:self.max_size]
            self._y = self._y[:self.max_size]
            self._meta = self._meta[:self.max_size]

        if self.flip_augment:
            self._meta = np.concatenate(
                [
                    self._meta, 
                    np.ones_like(self._meta, dtype=np.int8),
                    np.ones_like(self._meta, dtype=np.int8) + 1,
                    np.ones_like(self._meta, dtype=np.int8) + 2
                ],
                axis=0
            )
            self._x = np.tile(self._x, reps=4)
            self._y = np.tile(self._y, reps=4)

        # setting seed
        # deterministic shuffle ds
        self.shuffle()

    def __len__(self):
        return len(self._x)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._x[item], self._y[item], self._meta[item]

        elif isinstance(item, slice):
            return self._x[item], self._y[item], self._meta[item]
        else:
            raise TypeError("Dataset indices must be integers or slices, not {}.".format(type(item)))

    def __iter__(self):
        self._idx = 0
        # random shuffle dataset on stop iteration
        self.shuffle()
        return self

    def __next__(self):
        ds_len = len(self)
        # stop iteration condition
        if self._idx >= ds_len:
            raise StopIteration()

        x = self.read_img(self._x[self._idx])
        # labels
        y = self.read_img(self._y[self._idx])

        # flip augmentation on demand
        x, y = self.augment(self._meta[self._idx], x, y)

        # if mask_channels is bigger than one, then it will have exactly two channes
        # composed by the mask and its complement.
        if self.mask_channels > 1:
            y = np.concatenate([y, 1 - y], axis=-1)

        # go to next idx + 1
        self._idx += 1

        return x, y

    def build_dataset(self):
        h_resolution = int(self.config.get("dataset.h_resolution").strip())
        w_resolution = int(self.config.get("dataset.w_resolution").strip())
        output_shapes = (
            tf.TensorShape(
                [h_resolution, w_resolution, 3]
            ),  # x
            tf.TensorShape(
                [h_resolution, w_resolution, 2]
            )
        )

        output_types = (
            tf.float32,
            tf.float32
        )

        # build output type list
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: iter(self),
            output_types=output_types,
            output_shapes=output_shapes
        )
        self.dataset = dataset.batch(
            self.batch_size,
            drop_remainder=True
        ).prefetch(10)

        return self.dataset

    def read_img(self, path):
        image = Image.open(path)
        if "_mask.png" in path:
            image = image.convert('L')
            image = image.resize((self.patch_size[1], self.patch_size[0]))
            x = np.asarray(image, dtype=np.int32)
            x = np.expand_dims(x, axis=-1)
            x = x // 255
            x = x.astype(np.float32)
        else:
            image = image.convert('RGB')
            image = image.resize((self.patch_size[1], self.patch_size[0]))
            x = np.asarray(image, dtype=np.float32)
            # x /= 255

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

    def distribute_dataset(self, strategy):
        self.dataset = strategy.experimental_distribute_dataset(
            self.dataset
        )

    @property
    def current(self):
        idx = self._idx - 1
        return self._x[idx], self._y[idx]

    def shuffle(self):
        perm = self.random_state.permutation(len(self._x))
        self._x = self._x[perm]
        self._y = self._y[perm]
        self._meta = self._meta[perm]