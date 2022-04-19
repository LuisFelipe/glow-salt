# -*- coding: utf-8 -*-
"""
module patcher.py
--------------------
Tasks to pre-compute the dataset of salt patches from the model of salt repository.  
"""
from invoke import task
from main import Main
import os
import json
import tensorflow as tf
import sys
import utils
from utils import clear_callbacks
from datasets.raw_data_reader import RawDataReader
import numpy as np
# sys.path.append('..')


def _get_dataset(config, partition):
    return RawDataReader(
        config=config,
        model_path=config.get("dataset.{}.model_path".format(partition)),
        velocity_path=config.get("dataset.{}.velocity_path".format(partition)),
        model_size=[int(x) for x in config.get("dataset.{}.model_size".format(partition), "1040,7760").strip().split(",")],
        strip=[int(x) for x in config.get("dataset.{}.strip".format(partition)).strip().split(",")],
        striph=[int(x) for x in config.get("dataset.{}.striph".format(partition)).strip().split(",")],
        mask_threshold=float(config.get("dataset.{}.mask_threshold".format(partition))),
        patch_size=[int(x) for x in config.get("dataset.patch_size", "64,64").strip().split(",")],
        partition=partition.replace("2",""),
        partition_bound=[int(x) for x in config.get("dataset.{}.partition_bound".format(partition)).strip().split(",")],
        bounded=config.get("dataset.{}.bounded".format(partition), "false").strip().lower() == "true"
    )


def _save(path, partition, ds, n):
    count = 0

    for x, y in ds:
        i = ds._idx - 1
        idx_x, idx_y = ds._x[i], ds._y[i]
        
        utils.save_figure_from_array(
            os.path.join(path, "{}/{}_{}_{}.png".format(partition, idx_x, idx_y, n)),
             x * 255
        )
        utils.save_figure_from_array(
            os.path.join(path, "{}/{}_{}_{}_mask.png".format(partition, idx_x, idx_y, n)),
             y * 255
        )
        count += 1


def _save_patches(config):
    path = config.get("dataset.save_path", "../data/dataset/")
    utils.mkdir_if_not_exists(path)
    utils.mkdir_if_not_exists(os.path.join(path, "train/"))
    utils.mkdir_if_not_exists(os.path.join(path, "test/"))
    utils.mkdir_if_not_exists(os.path.join(path, "valid/"))

    # train partition
    ds1 = _get_dataset(config, "train")
    print(len(ds1._x), len(ds1._y))
    print(ds1._raw_model.shape)
    _save(path, "train", ds1, 1)

    ds1 = _get_dataset(config, "train2")
    print(len(ds1._x), len(ds1._y))
    print(ds1._raw_model.shape)
    _save(path, "train", ds1, 2)

    # valid partition
    ds2 = _get_dataset(config, "valid")
    print(len(ds2._x), len(ds2._y))
    print(ds2._raw_model.shape)
    _save(path, "valid", ds2, 3)

    ds2 = _get_dataset(config, "valid2")
    print(len(ds2._x), len(ds2._y))
    print(ds2._raw_model.shape)
    _save(path, "valid", ds2, 4)

    # test partition
    ds3 = _get_dataset(config, "test")
    print(len(ds3._x), len(ds3._y))
    print(ds3._raw_model.shape)
    _save(path, "test", ds3, 6)

    ds3 = _get_dataset(config, "test2")
    print(len(ds3._x), len(ds3._y))
    print(ds3._raw_model.shape)
    _save(path, "test", ds3, 5)


@task
def save_patches(ctx, config):
    """Computes and saves the salt model patches from the raw salt model. 
    
    Cuts the model in patches and save as png images separating the train, valid and test splits.
    :param ctx: the program conext.
    :param config: the configuration file path.
    :type config: str
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )
    main = Main(config_path=config)
    _save_patches(main.config)


def _filter_ds(config, ds, partition):
    save_path = config.get("dataset.save_path")
    max_salt_percent = float(config.get("dataset.max_salt_percent"))
    min_salt_percent = float(config.get("dataset.min_salt_percent"))
    
    _iter = iter(ds)
    _xy = filter(
        lambda xy: max_salt_percent >= ((np.sum(xy[1]) + 1)/(np.prod(xy[1].shape))) >= min_salt_percent, 
        _iter
    )

    count = 0
    for x, y in _xy:
        utils.save_figure_from_array(
            os.path.join(save_path, "{}/{}".format(partition, str(count) + ".png")),
            x
        )
        utils.save_figure_from_array(
            os.path.join(save_path, "{}/{}".format(partition, str(count) + "_mask.png")),
            y * 255
        )
        count += 1

@task
def filter_data(ctx, config):
    """ filter data by salt percentage.

    :param ctx: the program conext.
    :param config: the configuration file path.
    :type config: str
    """
    from datasets.salt_dsv2 import SaltDS
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    main = Main(config_path=config)
    path = main.config.get("dataset.save_path")
    utils.mkdir_if_not_exists(path)
    utils.mkdir_if_not_exists(os.path.join(path, "train/"))
    utils.mkdir_if_not_exists(os.path.join(path, "test/"))
    utils.mkdir_if_not_exists(os.path.join(path, "valid/"))

    
    ds_path = main.config.get("dataset.path")
    train_ds = SaltDS(
        config=main.config,
        path=ds_path,
        partition="train"
    )
    _filter_ds(main.config, train_ds, partition="train")
    del train_ds

    valid_ds = SaltDS(
        config=main.config,
        path=ds_path,
        partition="valid"
    )
    _filter_ds(main.config, valid_ds, partition="valid")
    del valid_ds

    test_ds = SaltDS(
        config=main.config,
        path=ds_path,
        partition="train"
    )
    _filter_ds(main.config, test_ds, partition="test")
    del test_ds

@task
def ds_to_tfrecord(ctx, config=''):
    """
    Convert the dataset to tfRecord format.

    :param config: Configuration file path.
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    main = Main(config_path=config)
    main.convert_to_tfrecord()