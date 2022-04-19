# -*- coding: utf-8 -*-
"""
module salt_ds.py
--------------------
salt dataset reader and iterator.
"""
import os
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image


class MaskDS(object):
    """ Salt deposits and masks dataset. 
    Reads the dataset and iterates over it.
    """

    def __init__(
        self, config=None, 
        path=None, partition="train", iterator=None, 
        *args, **kwargs
    ):
        """Main module initialization."""
        self._idx = None
        self._y = list()

        self.output_names = ("y",)
        self.partition = partition
        self.path = path
        self.config = config
        self.random_state = np.random.RandomState(123456)
        
        self.batch_size = int(self.config.get("FLOW.BATCH_SIZE", 1))
        self.prefetch_buffer = int(self.config.get("DATASET.BUFFER_SIZE", 1))

        # load img paths
        self.load_data()
        self.dataset = None
        self.build_dataset()

    def load_data(self):
        max_size = int(self.config.get("dataset.max_size", -1))

        for root, _, files in os.walk(self.path):
            for file in files:
                if not file.endswith(".png"):
                    continue
                p = os.path.join(root, file)
                self._y.append(p)

        # as np array
        self._y = np.asarray(self._y)

        # if dataset.max_size is set
        # then we truncate the dataset length to fit the max_size
        if max_size > 0:
            self._y = self._y[:max_size]

    def __len__(self):
        return len(self._y)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._y[item]

        elif isinstance(item, slice):
            return self._y[item]
        else:
            raise TypeError(
                "Dataset indices must be integers or slices, not {}.".format(type(item))
            )

    def __iter__(self):
        self._idx = 0
        # random shuffle dataset on stop iteration
        perm = self.random_state.permutation(len(self._y))
        self._y = self._y[perm]
        return self

    def __next__(self):
        ds_len = len(self)
        
        # stop iteration condition
        if self._idx >= ds_len:
            raise StopIteration()

        y = self.read_img(self._y[self._idx])
        y = np.expand_dims(y, axis=-1)

        # if mask_channels is bigger than one, then it will have exactly two channes
        # composed by the mask and its complement.
        y = np.concatenate([y, 1 - y], axis=-1).astype(np.float32)
        x = np.zeros(
            shape=[y.shape[0], y.shape[1], 3],
            dtype=np.float32
        )
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
        ).prefetch(self.prefetch_buffer)

        return self.dataset

    def read_img(self, path):
        H = int(self.config.get("dataset.h_resolution", 256))
        W = int(self.config.get("dataset.w_resolution", 192))

        image = Image.open(path)
        image = image.convert('L')

        image = image.resize((W, H), resample=PIL.Image.BICUBIC)
        x = np.asarray(image, dtype=np.float32)
        x /= 255
        # x = np.expand_dims(x, axis=2)

        # if self.mask_channels > 1:
        #     x = np.concatenate([x, 1 - x], axis=-1)

        return x
