# -*- coding: utf-8 -*-
"""
module salt_dsv2.py
--------------------
salt dataset reader and iterator second version.
It reads the dataset from a preprocessed data path.
Additionally, it can be used to augment the data with horizontal flips.
"""
import os
import glob
import tensorflow as tf
import numpy as np


class TFRecordDS(object):
    """Seismogram dataset iterator.
    Reads the dataset and iterates over it.
    """

    def __init__(self, config=None, path=None, partition="train", *args, **kwargs):
        """Main module initialization."""
        self.output_names = ("x", "y")

        self.partition = partition
        self.path = path
        self.config = config
        self._init_configs()
        
        # load img paths
        if partition == "all":
            self.partition = "train"
        self.dataset = None
        self.build_dataset()

    def __len__(self):
        return self._len

    def _init_configs(self):
        self.batch_size = int(self.config.get("flow.global_batch_size", 1))
        self.prefetch_buffer = int(self.config.get("DATASET.BUFFER_SIZE", 1))
        self.patch_size = [
            int(self.config.get("dataset.h_resolution")), 
            int(self.config.get("dataset.w_resolution"))
        ]
        self.mask_channels = int(self.config.get("dataset.mask_channels", 2))
        self.n_parallel_reads = int(self.config.get("dataset.n_parallel_reads", "1"))
        self.max_size = int(self.config.get("dataset.max_size", "-1"))

    def build_dataset(self):
        files = self.get_files()

        dataset = tf.data.TFRecordDataset(
            filenames=files,
            num_parallel_reads=self.n_parallel_reads
        ).map(
            map_func=self._parse_record, 
            num_parallel_calls=self.n_parallel_reads
        )
        
        if self.max_size > 0:
            dataset = dataset.take(self.max_size)

        self._len = sum([1 for x, y in dataset])

        self.dataset = dataset.shuffle(len(self)).batch(
            self.batch_size,
            drop_remainder=True
        ).prefetch(self.prefetch_buffer)

        return self.dataset

    def distribute_dataset(self, strategy):
        self.dataset = strategy.experimental_distribute_dataset(
            self.dataset
        )

    def _parse_record(self, tf_example):
        # maping tf record content to dictionary object
        features = {
            'image/encoded':
                tf.io.FixedLenFeature(
                    shape=(), 
                    dtype=tf.string, 
                    default_value=''
                ),
            'image/filename':
                tf.io.FixedLenFeature(
                    shape=(), 
                    dtype=tf.string, 
                    default_value=''
                ),
            'image/format':
                tf.io.FixedLenFeature(
                    shape=(), 
                    dtype=tf.string, 
                    default_value='jpeg'
                ),
            'image/height':
                tf.io.FixedLenFeature(
                    shape=(), 
                    dtype=tf.int64, 
                    default_value=0
                ),
            'image/width':
                tf.io.FixedLenFeature(
                    shape=(), 
                    dtype=tf.int64, 
                    default_value=0
                ),
            'image/segmentation/class/encoded':
                tf.io.FixedLenFeature(
                    shape=(), 
                    dtype=tf.string, 
                    default_value=''
                ),
            'image/segmentation/class/format':
                tf.io.FixedLenFeature(
                    shape=(), 
                    dtype=tf.string, 
                    default_value='png'
                ),
        }

        parsed_features = tf.io.parse_single_example(tf_example, features)

        # parse x sample
        x = tf.io.parse_tensor(
            parsed_features['image/encoded'], 
            out_type=tf.float32
            # out_type=tf.double
        )
        x = tf.reshape(x, shape=(self.patch_size[0], self.patch_size[1], 3))

        # parse y mask sample
        y = tf.io.parse_tensor(
            parsed_features['image/segmentation/class/encoded'], 
            out_type=tf.float32
        )
        y = tf.reshape(
            y, 
            shape=(self.patch_size[0], self.patch_size[1], 2)
        )

        return x, y

    def get_files(self):
        file_list = tf.io.gfile.glob(
            os.path.join(self.path, "{}-*.tfrecord".format(self.partition))
        )
        print("**********", file_list, "**************")
        return file_list
