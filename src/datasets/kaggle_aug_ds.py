# -*- coding: utf-8 -*-
"""
module kaggle_aug_ds.py
--------------------
salt dataset reader and iterator second version.
It reads the dataset from a preprocessed data path.
Additionally, it can be used to augment the data with horizontal flips.
"""
import os
import tensorflow as tf
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage


class KaggleAugDS(object):
    """Seismogram dataset iterator.
    Reads the dataset and iterates over it.
    """

    def __init__(self, config=None, path=None, partition="train", *args, **kwargs):
        """Main module initialization."""
        self._idx = None
        self._meta = None
        self._x = list()
        self._y = list()
        # self._x_augs = list()
        # self._y_augs = list()
        self.output_names = ("x", "y")

        # self.partition = partition
        self.path = path
        self.config = config
        self.cmap = plt.get_cmap("seismic")
        self.random_state = np.random.RandomState(1234567)

        self._init_configs()
        # load img paths
        self.load_data()
        if partition == "all":
            partition = "train"
        self.partition = partition
        self.dataset = None
        self.build_dataset()

    def _init_configs(self):
        self.flip_augment = self.config.get("dataset.augment", "false").strip().lower() == "true"
        self.max_size = int(self.config.get("dataset.max_size", -1))
        self.batch_size = int(self.config.get("flow.global_batch_size", 1))
        self.prefetch_buffer = int(self.config.get("DATASET.BUFFER_SIZE", 1))
        self.patch_size = [int(self.config.get("dataset.h_resolution")), int(self.config.get("dataset.w_resolution"))]
        self.valid_size = float(self.config.get("dataset.valid_size", 0.1))
        self.mask_channels = int(self.config.get("dataset.mask_channels", 2))

    def load_data(self):
        path = self.path
        

        # load partition
        for root, dirs, files in os.walk(path):
            for f in files:
                if not f.endswith(".png"):
                    continue
                if "mask_" in f:
                    continue
                if not os.path.exists(os.path.join(root, f.replace("image_", "mask_"))):
                    continue

                f_path = os.path.join(root, f)
                self._x.append(f_path)
                f_path = os.path.join(root, f.replace("image_", "mask_"))
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
                    np.ones_like(self._meta, dtype=np.int8) + 2,
                    np.ones_like(self._meta, dtype=np.int8) + 3,
                    np.ones_like(self._meta, dtype=np.int8) + 4,
                    np.ones_like(self._meta, dtype=np.int8) + 5,
                ],
                axis=0
            )
            self._x = np.tile(self._x, reps=6)
            self._y = np.tile(self._y, reps=6)

        # setting seed
        # deterministic shuffle ds
        self._shuffle()

    def __len__(self):
        return len(self._x)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._x[item], self._y[item], self._meta[item]
            # return self._x[item], self._y[item]

        elif isinstance(item, slice):
            return self._x[item], self._y[item], self._meta[item]
            # return self._x[item], self._y[item]
        else:
            raise TypeError("Dataset indices must be integers or slices, not {}.".format(type(item)))

    def __iter__(self):
        self._idx = 0
        # random shuffle dataset on stop iteration
        self._shuffle()
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
        if "mask_" in path:
            image = image.convert('L')
            image = image.resize((self.patch_size[1], self.patch_size[0]))
            x = np.asarray(image, dtype=np.int32)
            x = np.expand_dims(x, axis=-1)
            x = x // 255
            x = x.astype(np.float32)
        else:
            # image = image.convert('L')
            image = image.resize((self.patch_size[1], self.patch_size[0]))
            x = np.asarray(image, dtype=np.float32)
            x /= 255
            x = x[:,:,0]
            x = self.cmap(x)[:, :, :3]
            x *= 255
            x = x.astype(np.float32)

        return x

    def augment(self, augmentation_type, x, y):

        # x = ndimage.shift(x, shift=[0, -10,0], mode='reflect', order=3)
        # y = ndimage.shift(y, shift=[0, -10,0], mode='reflect', order=3)
        # x = ndimage.shift(x, shift=[0, 10,0], mode='reflect')
        # y = ndimage.shift(y, shift=[0,10,0], mode='reflect')            
        if augmentation_type == 1:
            x = ndimage.shift(x, shift=[0, 10, 0], mode='reflect')
            y = ndimage.shift(y, shift=[0,10, 0], mode='reflect')
        elif augmentation_type == 2:
            x = ndimage.shift(x, shift=[10, 0,0], mode='reflect', order=3)
            y = ndimage.shift(y, shift=[10, 0,0], mode='reflect', order=3)
        elif augmentation_type == 3:
            x = ndimage.shift(x, shift=[10, 10,0], mode='reflect', order=3)
            y = ndimage.shift(y, shift=[10, 10,0], mode='reflect', order=3)
        elif augmentation_type == 4:
            x = ndimage.shift(x, shift=[0, -10, 0], mode='reflect')
            y = ndimage.shift(y, shift=[0, -10, 0], mode='reflect')
        elif augmentation_type == 5:
            x = ndimage.shift(x, shift=[-10, 0,0], mode='reflect', order=3)
            y = ndimage.shift(y, shift=[-10, 0,0], mode='reflect', order=3)
        elif augmentation_type == 6:
            x = ndimage.shift(x, shift=[-10, -10,0], mode='reflect', order=3)
            y = ndimage.shift(y, shift=[-10, -10,0], mode='reflect', order=3)
        
        return x, y

    def distribute_dataset(self, strategy):
        self.dataset = strategy.experimental_distribute_dataset(
            self.dataset
        )

    @property
    def current(self):
        idx = self._idx - 1
        return self._x[idx], self._y[idx]

    def _shuffle(self):
        perm = self.random_state.permutation(len(self._x))
        self._x = self._x[perm]
        self._y = self._y[perm]
        self._meta = self._meta[perm]