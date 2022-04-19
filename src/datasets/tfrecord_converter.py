# -*- coding: utf-8 -*-
"""
module tfrecord_converter.py
--------------------------------
    Converts the specified dataset split to TFRecord format.
""" 
import os
import os.path
from flow.config import Config
import tensorflow as tf
import collections
import utils


class DsConverter(object):
    """ Converts the  dataset to a TFRecord format.
    
    Converts image dataset to a TFRecord format.
    """
    def __init__(self, dataset, num_shards=2):
        self.config = Config()
        self.img_size = [
            int(self.config.get("dataset.h_resolution")),
            int(self.config.get("dataset.w_resolution"))
        ]
        self.output_path = self.config.get("dataset.tfrecord_path")
        self.dataset = dataset
        self.num_shards = num_shards

    def _convert_dataset(self):
        """Converts the specified dataset split to TFRecord format.

        :param dataset_split: The dataset split (e.g., train, test).

        :raise RuntimeError: If loaded image and label have different shape.
        """
        
        _reader = iter(self.dataset)
        
        images_per_shard = int(
                len(self.dataset) // self.num_shards
        )
        print("*******", images_per_shard, "*********")
        print("*******", len(self.dataset), "*********")
        # for each tfRecord shard
        for shard_id in range(self.num_shards):
            output_filename = os.path.join(
                self.output_path,
                '{}-{:05d}-of-{:05d}.tfrecord'.format(
                    self.dataset.partition, 
                    shard_id, self.num_shards
                )
            )
            with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:

                for i in range(images_per_shard):
                    print('\r>> Converting image {:d}/{:d} shard {:d} split {}'.format(
                            i + 1, 
                            images_per_shard, 
                            shard_id,
                            self.dataset.partition
                        ), end="\r"
                    )

                    # Read the image.
                    image_data, seg_data = next(_reader)
                    image_path, seg_path = self.dataset.current
                    image_path = image_path.split("/")[-1]

                    height, width = self.img_size
                    seg_height, seg_width = self.img_size
                    # Convert to tf example.
                    example = self.to_tfexample(
                        tf.io.serialize_tensor(image_data).numpy(), 
                        image_path, 
                        height, 
                        width, 
                        tf.io.serialize_tensor(seg_data).numpy()
                    )

                    tfrecord_writer.write(example.SerializeToString())
            print('')

    def convert(self):
        utils.mkdir_if_not_exists(self.output_path)
        self._convert_dataset()

    def _int64_list_feature(self, values):
        """Returns a TF-Feature of int64_list.

        Args:
          values: A scalar or list of values.

        Returns:
          A TF-Feature.
        """
        if not isinstance(values, collections.Iterable):
            values = [values]

        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def _bytes_list_feature(self, values):
        """Returns a TF-Feature of bytes.

        :param values: A string.

        :return: A TF-Feature.
        """
        def norm2bytes(value):
            return value.encode() if isinstance(value, str) else value

        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[norm2bytes(values)])
        )

    def to_tfexample(self, image_data, filename, height, width, seg_data):
        """Converts one image/segmentation pair to tf example.

        :param image_data: string of image data.
        :param filename: image filename.
        :param height: image height.
        :param width: image width.
        :param seg_data: string of semantic segmentation data.

        :return: tf example of one image/segmentation pair.
        """
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded': self._bytes_list_feature(image_data),
                    'image/filename': self._bytes_list_feature(filename),
                    'image/format': self._bytes_list_feature("png"),
                    'image/height': self._int64_list_feature(height),
                    'image/width': self._int64_list_feature(width),
                    'image/channels': self._int64_list_feature(3),
                    'image/segmentation/class/encoded': (
                        self._bytes_list_feature(seg_data)
                    ),
                    'image/segmentation/class/format': self._bytes_list_feature("png"),
                }
            )
        )
