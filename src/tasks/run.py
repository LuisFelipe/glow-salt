from invoke import task
from main import Main
import os
import json
import tensorflow as tf
import sys
from utils import clear_callbacks

# sys.path.append('..')


@task
def train(ctx, data=None, config='', data_init=True):
    """
    Function to Train the Classifier.

    Receives the dataset path and a classification file, train the model over the labels presented
    in the classification file.

    :param data: path to the desired data folder
    :param labels:  path to the classification file.
        The classification file contains the informations from the filename and class to each sismogram.
    :return: save the predictions in a given file
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )
    main = Main(config_path=config)
    main.train()


@task
def conditional_sample(ctx, data=None, config=''):
    """
    Function to Train the Classifier.

    Receives the dataset path and a classification file, train the model over the labels presented
    in the classification file.

    :param data: path to the desired data folder
    :param labels:  path to the classification file.
        The classification file contains the informations from the filename and class to each sismogram.
    :return: save the predictions in a given file
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    main = Main(config_path=config)
    main.conditional_sample() 

@task
def print_history(ctx, data=None, config=''):
    """
    Function to Train the Classifier.

    Receives the dataset path and a classification file, train the model over the labels presented
    in the classification file.

    :param data: path to the desired data folder
    :param labels:  path to the classification file.
        The classification file contains the informations from the filename and class to each sismogram.
    :return: save the predictions in a given file
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    main = Main(config_path=config)
    main.print_history()

@task
def print_data(ctx, config=''):
    """
    Function to Train the Classifier.

    Receives the dataset path and a classification file, train the model over the labels presented
    in the classification file.

    :param data: path to the desired data folder
    :param labels:  path to the classification file.
        The classification file contains the informations from the filename and class to each sismogram.
    :return: save the predictions in a given file
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    cfg_name = config.split("/")[-1]
    cfg_dir = config.replace(cfg_name, "")
    # model fit loading initialized weights
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        main = Main(tf.get_default_session(), config_name=cfg_name, config_path=cfg_dir)
        main.print()

@task
def encode_decode(ctx, config=''):
    """
    Function to Train the Classifier.

    Receives the dataset path and a classification file, train the model over the labels presented
    in the classification file.

    :param data: path to the desired data folder
    :param labels:  path to the classification file.
        The classification file contains the informations from the filename and class to each sismogram.
    :return: save the predictions in a given file
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    cfg_name = config.split("/")[-1]
    cfg_dir = config.replace(cfg_name, "")
    # model fit loading initialized weights
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        main = Main(tf.get_default_session(), config_name=cfg_name, config_path=cfg_dir)
        main.encode_decode()

@task
def eval_loss(ctx, config=''):
    """
    Function to Train the Classifier.

    Receives the dataset path and a classification file, train the model over the labels presented
    in the classification file.

    :param data: path to the desired data folder
    :param labels:  path to the classification file.
        The classification file contains the informations from the filename and class to each sismogram.
    :return: save the predictions in a given file
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    main = Main(config_path=config)
    main.eval_loss() 
