# -*- coding: utf-8 -*-
"""
module model.py
--------------------
Definition of the machine learning model for the task.
"""
import os
import tensorflow as tf


class RunTimeConfig(object):
    def __init__(self, main):
        self.main = main
        self.config = self.main.config
        self.device_type = self.config.get("flow.hardware_accelerator").upper()
        self.strategy = None

    def configure_device(self):
        # enable eager execution in tensorflow 1.x versions
        if tf.__version__.startswith("1."):
            tf.compat.v1.enable_eager_execution()

        if self.device_type == "GPU":
            self._configure_gpu()

        elif self.device_type == "TPU":
            self._configure_tpu()

        elif self.device_type == "CPU":
            self._configure_cpu()

        batch_size = int(self.config.get("flow.batch_size"))
        global_batch_size = batch_size * self.strategy.num_replicas_in_sync
        print("***** n_replicas: ", self.strategy.num_replicas_in_sync, "*******")
        self.config["flow.global_batch_size"] = global_batch_size

    def _configure_cpu(self):
        physical_devices = tf.config.experimental.list_physical_devices('CPU')
        cpu_name = physical_devices[0].name.replace("physical_device:", "")
        self.strategy = tf.distribute.OneDeviceStrategy(device=cpu_name)

    def _configure_gpu(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        device_ids = [
            int(device_id.strip()) 
            for device_id in self.config.get("flow.visible_devices", "").split(",")
        ]

        print("*******Physical Devices************", "\n\t", physical_devices)

        if len(device_ids) > 0:
            devices = [
                physical_devices[device_id] for device_id in device_ids
            ]
        else:
            devices = physical_devices

        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)

        tf.config.set_visible_devices(devices, 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        assert len(logical_devices) == len(devices)

        if len(devices) == 1:
            print(">>>>>>>", devices[0])
            gpu_name = devices[0].name.replace("physical_device:", "")
            self.strategy = tf.distribute.OneDeviceStrategy(device=gpu_name)
        else:
            gpus = [d.name.replace("physical_device:", "") for d in devices]
            self.strategy = tf.distribute.MirroredStrategy(devices=gpus)

    def _configure_tpu(self):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ['COLAB_TPU_ADDR']
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        self.strategy = tf.distribute.experimental.TPUStrategy(resolver)

    def get_distribution_strategy(self):
        if self.strategy is None:
            self.configure_device()
        return self.strategy
