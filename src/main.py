# -*- coding: utf-8 -*-
"""
module model.py
--------------------
Definition of the machine learning model for the task.
"""
import os
# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# setting visible devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Use tensorcores
# enable tf mixed precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926
# os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL) 
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.get_logger().setLevel("ERROR")
import numpy as np
from runtime_config import RunTimeConfig
from flow.config import Config
from models import InvertibleNet, SegFlowBert
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")


class Main(object):
    __config_path__ = ""

    def __init__(self, config_path=None):
        self.config = Config()
        self.config.add_path(config_path)
        self.config.load_config()
        self._runtime_config = RunTimeConfig(self)
        self.strategy = self._runtime_config.get_distribution_strategy()
        # tf.compat.v1.random.set_random_seed(54321)
        # np.random.seed(4325)

    def get_model(self, *args, **kwargs):
        name = self.config.get("model.name", default="SegFlow").upper()
        if name == "SEGFLOW":
            with self.strategy.scope():
                model = InvertibleNet()
                model.build()
                self.model = model
        elif name == "SEGFLOWBERT":
            with self.strategy.scope():
                model = SegFlowBert()
                model.build()
                self.model = model
        return model

    def get_dataset(self):
        ds_name = self.config.get("dataset.dataset_name", "salt_ds")
        if ds_name == "mnist":
            from datasets.mnist import MnistDS
            train_ds = MnistDS(
                self.inputs_config,
                config=self.config,
                path=None,
                partition="train"
            )
            valid_ds = MnistDS(
                self.inputs_config,
                config=self.config,
                path=None,
                partition="valid"
            )
            return train_ds, valid_ds

        elif ds_name == "salt_ds":
            from datasets.salt_ds import SaltDS
            ds_path = self.config.get("dataset.path", "")
            train_ds = SaltDS(
                config=self.config,
                path=ds_path,
                partition="train"
            )
            valid_ds = SaltDS(
                config=self.config,
                path=ds_path,
                partition="valid"
            )
            print(">>>>>ds_len>>>>>", len(train_ds))
            return train_ds, valid_ds

        elif ds_name == "salt_dsV2":
            from datasets.salt_dsv2 import SaltDS
            ds_path = self.config.get("dataset.path", "")
            train_ds = SaltDS(
                config=self.config,
                path=ds_path,
                partition="all"
            )
            valid_ds = SaltDS(
                config=self.config,
                path=ds_path,
                partition="valid"
            )
            print(">>>>>ds_path>>>>>", ds_path)
            print(">>>>>ds_len>>>>>", len(train_ds))
            train_ds.distribute_dataset(self.strategy)
            valid_ds.distribute_dataset(self.strategy)
            self.config["dataset.data_len"] = len(train_ds)

            return train_ds, valid_ds

        elif ds_name == "tfrecord_ds":
            from datasets.tfrecord_ds import TFRecordDS
            ds_path = self.config.get("dataset.tfrecord_path", "")
            train_ds = TFRecordDS(
                config=self.config,
                path=ds_path,
                partition="all"
            )
            valid_ds = TFRecordDS(
                config=self.config,
                path=ds_path,
                partition="all"
            )
            print(">>>>>ds_len>>>>>", len(train_ds))
            train_ds.distribute_dataset(self.strategy)
            valid_ds.distribute_dataset(self.strategy)
            self.config["dataset.data_len"] = len(train_ds)

            return train_ds, valid_ds

        elif ds_name == "mask_ds":
            from datasets.mask_ds import MaskDS
            ds_path = self.config.get("dataset.path", "")
            mask_ds = MaskDS(
                config=self.config,
                path=ds_path,
                partition="train"
            )
            _ = None
            self.config["dataset.data_len"] = len(mask_ds)
            return mask_ds, _

        elif ds_name == "kaggle_aug_ds":
            from datasets.kaggle_aug_ds import KaggleAugDS
            ds_path = self.config.get("dataset.path", "")
            train_ds = KaggleAugDS(
                config=self.config,
                path=ds_path,
                partition="all"
            )
            valid_ds = KaggleAugDS(
                config=self.config,
                path=ds_path,
                partition="valid"
            )
            print(">>>>>ds_len>>>>>", len(train_ds))
            train_ds.distribute_dataset(self.strategy)
            valid_ds.distribute_dataset(self.strategy)
            self.config["dataset.data_len"] = len(train_ds)

            return train_ds, valid_ds

    def train(self, *args, **kwargs):
        should_resume = self.config.get("flow.resume", "False").lower() == "true"

        with self.strategy.scope():
            train_ds, valid_ds = self.get_dataset()
            model = self.get_model()
            # if should_resume:
            #     p = self.config.get("flow.checkpoint")
            #     model.load(tf.train.latest_checkpoint(p))

            model.fit(
                train_dataset=train_ds,
                valid_dataset=None,
                resume=should_resume
            )
        return model

    def conditional_sample(self):
        from PIL import Image
        # data dependent initialization must be done in non distributed setting because of tf graph mode issues
        n_samples = int(self.config.get("samples.n_samples", 6000))
        base_path = self.config.get("samples.save_path")
        utils.mkdir_if_not_exists(base_path)

        ds_path = self.config.get("flow.checkpoint", "")
        ds_path = os.path.join(ds_path, "sample_masks/")

        mask_ds, _ = self.get_dataset()
        # mask_ds.shuffle()
        model = self.get_model()
        # build backwards graph to sample from the latent space.
        # model.backward_pass()
        

        count = 0
        for x_in, y in mask_ds.dataset:
            if count == 0:
                model(x_in, forward=True)
                model.load(
                    tf.train.latest_checkpoint(
                        self.config.get("flow.premodel")
                    )
                )

            outs = model(y, forward=False)
            # print(">>>>>>>>>>>>", tf.reduce_max(x_in))
            # raise ValueError()
            x, logdet_jacob, _eps = outs
            # print(">>>>", x.shape)
            # print(x[0])
            # print(">>>>", logdet_jacob.shape)
            # print(logdet_jacob[0])
            for j in range(x.numpy().shape[0]):
                fig, (ax1, ax2, ax3) = plt.subplots(
                    1, 3, 
                    figsize=(40, 20)
                )
                
                ax1.imshow(x[j]/255)
                ax1.set_title('Predicted')

                ax2.imshow(x_in[j]/255)
                ax2.set_title('Ground Truth')
                
                ax3.imshow(y[j][:, :, 0])
                ax3.set_title('Mask')
                plt.show()

                # utils.save_figure_from_array(
                #     path=os.path.join(base_path, "{}.png".format(count)),
                #     array=x.numpy()[j][:, :, :]
                # )
                # utils.save_figure_from_array(
                #     path=os.path.join(base_path, "{}_mask.png".format(count)),
                #     array=np.expand_dims(y.numpy()[j][:, :, 0], -1) * 255
                # )
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break

    def print_history(self):
        hist_path = self.config.get("flow.checkpoint")
        hist_path = os.path.join(hist_path, "_history.json")

        import matplotlib.pyplot as plt
        plt.style.use('seaborn-white')
        import seaborn as sns
        sns.set_style("white")
        import json

        with open(hist_path, mode="r") as fp:
            _json = json.load(fp)
        print(">>>######>>>", list(_json.keys()))
        
        elapsed_time = _json["epoch_elapsed_time_in_seconds"]
        # computing total time in h:m:s
        tot_seconds = sum(filter(lambda x: x is not None, elapsed_time))
        hours = (tot_seconds // 60) // 60
        minutes = (tot_seconds // 60) % 60
        seconds = tot_seconds % 60
        # print elapsed time h:m:s
        print("elapsed_time: 0:{}:{}".format(
                hours, 
                minutes,
                seconds
            )
        )

        bits_x = _json["loss/px_given_y"]
        print(len(bits_x))
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.set_xlabel("Epoch", fontsize=25)
        ax.set_ylabel("Bits per pixel.", fontsize=25)
        # ax.set(xlabel="Epoch", ylabel="Bits per pixel.")

        bits_x = list(filter(lambda x: x > 0 and not np.isnan(x), bits_x))
        ticks = list(range(0, len(bits_x), 200))
        ticks[0] = 1
        ax.xaxis.set_ticks(ticks)
        ax.plot(bits_x)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.show()

    def convert_to_tfrecord(self):
        from datasets.tfrecord_converter import DsConverter
        train_ds, valid_ds = self.get_dataset()

        converter = DsConverter(train_ds, num_shards=8)
        converter.convert()

    def eval_loss(self):
        import collection
        from tqdm import tqdm

        batch_size = int(self.config.get("flow.batch_size", default="10"))
        
        train_ds, _ = self.get_dataset()
        # mask_ds.shuffle()
        model = self.get_model()
        total = 0
        with tqdm(desc="Calculating Training Score:", total=len(train_ds)) as pbar:
            for count, (x_in, y) in enumerate(train_ds.dataset):

                for t, name in zip((x_in, y), train_ds.output_names):
                    collection.add_to_collection(
                        collection_name="inputs", 
                        tensor=t, 
                        tensor_name=name
                    )

                if count == 0:
                    outs = model(x_in, forward=True)
                    loss, metrics = model.loss_and_metrics(outs, y)
                    model.load(
                        tf.train.latest_checkpoint(
                            self.config.get("flow.premodel")
                        )
                    )

                outs = model(x_in,forward=True)
                loss, metrics = model.loss_and_metrics(outs, y)

                # clear all collections
                collection.clear_all()
                total += (metrics[0].numpy() / 500)
                # total.append(loss.numpy())
                pbar.update(batch_size)
                # print(loss.numpy() )
                print(metrics[0].numpy() )
                print(metrics[1].numpy() )

        # print("Model Loss:", np.mean(total))
        print("Model Loss:", total)