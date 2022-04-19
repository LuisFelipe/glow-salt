# -*- coding: utf-8 -*-
"""
module model.py
--------------------
Definition of the machine learning model for the task.
"""
import tensorflow as tf
import numpy as np
import utils
from flow.config import Config
from layers import Conv1x1
from layers import ActNorm, CouplingLayer, FactorOut
from layers import Squeeze
from losses.loss import Loss
from . import Prior
from .logits_net import LogitsNet
from flow.dataset import Dataset
from .learner import Learner
from .lr_scheduler import LrScheduler
import collection


class InvertibleNet(tf.keras.Model):
    """
    Generative Flow With Invertible neural network model class.
    """

    def __init__(self):
        """
        Model initialization function.
        :param inputs_config: a dictionary defining the shape and type of the model's inputs.
        """
        super(InvertibleNet, self).__init__()
        self.built = False
        self.config = Config()
        self.init_configs()
        self.optimizer = None
        self._blocks = list()

    def build(self):
        """
        Builds the model tensors and layers.
        It also sets the appropriate outputs variables.
        :return: a compiled model
        """
        if not self.built:
            self.init_layers()
            # call backs and optmizer
            self.get_optimizer()
            # update op
            self.get_callbacks()

            self.built = True

    def init_configs(self):
        """
        gets the model's configs and saves into class parameters.
        """
        # flow configs
        self.batch_size = int(self.config.get("flow.batch_size"))
        self.global_batch_size = int(self.config["flow.global_batch_size"])

        # model configs
        self.weight_y = float(self.config.get("model.weight_y", 0))
        self.depth = [int(x) for x in self.config.get("MODEL.DEPTH", "4,3,2").split(",")]
        self.levels = int(self.config.get("MODEL.LEVELS", 3))
        self.n_filters = int(self.config.get("MODEL.N_FILTERS", 64))
        self.n_bits_x = float(self.config.get("model.n_bits_x", 8))
        self.attention_levels = self.config.get("model.attention_levels", [])
        if len(self.attention_levels) > 0:
            self.attention_levels = [int(x) for x in self.attention_levels.split(",")]

        self.attention_block_type = self.config.get("model.attention_block_type", "block3")
        self.block_type = self.config.get("affine_layer.block_type", "block1")

        # optimizer configs
        # initial learning rate
        self.init_lr = float(self.config.get("optimizer.lr", 1e-5))
        self.per_epoch_steps = int(self.config.get("dataset.data_len")) // self.global_batch_size
        epochs_warmup = int(self.config.get("optimizer.warmup_epochs"))
        self.warmup_steps = self.per_epoch_steps * epochs_warmup

    def init_layers(self):
        # building reversible arquitecture.
        print(">>>depth<<<<<", self.depth)
        for l in range(self.levels):
            squeeze = [Squeeze(name="l.{}-Squeeze".format(l), factor=2)]
            self._blocks.append(squeeze)

            for i in range(self.depth[l]):
                block = self.block(self.n_filters, l, i)
                self._blocks.append(block)

            if l < self.levels - 1:
                factorout = [FactorOut(name="l.{}-FactorOut".format(l))]
                self._blocks.append(factorout)

        self.logits_layer = LogitsNet(name="logits_net")
        self._prior = Prior()
        self._loss = Loss(self.weight_y)

    def call(self, inputs, forward=True, training=False):
        if forward:
            x = inputs

            [z, logdet_jacob, _eps] = self.forward_pass(x)

            return z, logdet_jacob, _eps
        else:
            tf.keras.backend.set_learning_phase(False)
            mask = inputs
            outputs = self.backward_pass(mask=mask)
            return outputs

    def preprocess(self, x):
        n_bins = float(2. ** self.n_bits_x)
        # x = tf.cast(x, dtype=tf.float32, name="input_z")
        if self.n_bits_x < 8:
            x = (x / 2 ** (8 - self.n_bits_x))
            x = tf.floor(x)

        x /= n_bins
        return x - 0.5

    def postprocess(self, x):
        x_pred = x
        n_bins = float(2. ** self.n_bits_x)
        return tf.cast(
            tf.clip_by_value(
                tf.floor((x_pred + .5) * n_bins) * (256. / n_bins),
                # tf.floor((x_pred + .5) * self.n_bins),
                # tf.round((x_pred + .5) * 255),
                clip_value_min=0,
                clip_value_max=255
            ),
            dtype=tf.uint8
        )

    def forward_pass(self, x):
        """
        Model forward pass definition.
        :return: output tensors of the forward pass.
        """
        # preprocessing
        # x = self.inputs.x
        z = self.preprocess(x)
        # z = x
        z_shape = [self.batch_size] + z.shape.as_list()[1:]

        n_bins = float(2. ** self.n_bits_x)
        # from discrete -> to continuous

        z = z + tf.random.uniform(
            shape=z_shape,
            minval=0.,
            maxval=1. / n_bins
        )

        logdet_jacob = tf.zeros(shape=(self.batch_size,), dtype=tf.float32)
        logdet_jacob += - np.log(n_bins) * np.prod(z_shape[1:])

        # network inputs
        outputs = [
            z,
            logdet_jacob
        ]

        for block in self._blocks:
            for l in block:
                outputs = l(outputs, forward=True)
        
        # work arround to get _eps.shape correctly
        self._eps_shape = outputs[2].shape.as_list()
        return outputs

    def backward_pass(self, mask=None):
        """
        Model backward pass definition.
        :return: output tensors of the backwards pass.
        """
        # inputs
        _temp = float(self.config.get("samples.temperature", 1.0))
        _mean = float(self.config.get("samples.mean", 0))
        # prior
        mean, log_std = self._prior(
            [
                mask, 
                tf.zeros(
                    shape=(
                        self.batch_size,
                        mask.shape[1] // (2**self.levels),
                        mask.shape[1] // (2**self.levels),
                        3 * (2 ** (self.levels + 1))
                    ), 
                    dtype=tf.float32
                )
            ]
        )

        z = tf.random.normal(
            tf.shape(mean),
            mean=_mean,
            stddev=_temp,
            dtype=tf.float32
        )
        z = mean + tf.exp(log_std) * z

        _eps = tf.random.normal(
            shape=(
                self.batch_size,
                mask.shape[1] // (2**self.levels),
                mask.shape[1] // (2**self.levels),
                # mean.shape[-1] * ((2 * self.levels) - 1)
                self._eps_shape[-1]
            ),
            mean=_mean,
            stddev=_temp,
            dtype=tf.float32
        )
        print(">>>>>>>>sample_eps_shape<<<<<<<<<", _eps.shape)
        logdet_jacob = tf.zeros(
            shape=(self.batch_size,),
            dtype=tf.float32
        )
        
        outputs = [
            z,
            logdet_jacob,
            _eps
        ]
        # BACKWARD PAS (REVERSED PASS)
        for block in reversed(self._blocks):
            for l in reversed(block):
                outputs = l(outputs, forward=False)

        [x, logdet_jacob, _eps] = outputs
        x = self.postprocess(x)
        outputs = [x, logdet_jacob, _eps]
        return outputs

    def block(self, filters, layer_number, block_number):
        """ A single block of the invertible net.
        :param filters: number of filters used by the layers insede the block.
        :param layer_number: number of the current layer in the sequence.
        :param block_number: number of the current block in the sequence of layers block.
        :return: a invertible net block definition.
        """
        block = list()

        block.append(
            ActNorm(name="l.{}/b.{}_ActNorm".format(layer_number, block_number))
        )
        block.append(
            Conv1x1(
                name="l.{}/b.{}_Conv1x1".format(layer_number, block_number)
            )
        )

        if layer_number in self.attention_levels:
            block.append(
                CouplingLayer(
                    filters=filters,
                    level_num=layer_number,
                    block_num=block_number,
                    block_type=self.attention_block_type,
                    name="l.{}/b.{}_Affine".format(layer_number, block_number),
                )
            )
        else:
            block.append(
                CouplingLayer(
                    filters=filters,
                    level_num=layer_number,
                    block_num=block_number,
                    block_type=self.block_type,
                    name="l.{}/b.{}_Affine".format(layer_number, block_number),
                )
            )
        return block

    def loss_and_metrics(self, predicted, target):
        """
        Loss definition.
        """
        [z, logdet_jacob, _eps] = predicted
        y_mask = target

        mean, log_std = self._prior([y_mask, z])
        y_logits = self.logits_layer(z)

        x = collection.get_tensor_from_collection(
            collection_name="inputs",
            tensor_name="x"
        )
        # build loss _inputs list
        _inputs = [
            x,
            y_mask,
            logdet_jacob,
            z,
            _eps,
            y_logits,
            mean,
            log_std
        ]
        _loss = self._loss(_inputs)

        # y_logits = tf.nn.sigmoid(y_logits)
        y_pred = tf.cast(tf.greater_equal(y_logits, 0.5), dtype=tf.int32)
        y_mask = tf.cast(tf.greater_equal(y_mask, 0.5), dtype=tf.int32)
        acc = tf.cast(
            tf.equal(
                y_pred[:, :, :, 0], 
                y_mask[:, :, :, 0]
            ), 
            tf.float32
        )
        acc = tf.reduce_mean(
            acc, 
            axis=[1, 2]
        )
        acc = tf.nn.compute_average_loss(
            acc,
            global_batch_size=self.global_batch_size
        )
        acc._name = "accuracy"
        return _loss[0], _loss[1:] + [acc]

    def get_callbacks(self):
        # from flow.callbacks.early_stop import EarlyStopping, ModeEnum
        # from flow.callbacks.checkpointer import CheckPointer, ModeEnum
        from callbacks.checkpointer import CheckPointer, ModeEnum
        from callbacks.history import History
        from callbacks.timers import Timers
        from callbacks import ReduceLROnPlateau
        # from callbacks.before_epoch import BeforeEpoch

        # clear callbacks to avoid multiple runs of the same callback
        from utils import clear_callbacks
        clear_callbacks()

        self._checkpointer = CheckPointer(
            model=self,
            path=self.config.get("flow.checkpoint"),
            monitor="loss",
            verbose=1,
            save_best_only=False,
            mode=ModeEnum.MIN,
            max_to_keep=int(self.config.get("flow.max_checkpoints_to_keep", 3))
        )
        # self._reduce_lr = ReduceLROnPlateau(
        #     model=self, 
        #     factor=0.5, 
        #     monitor='loss',
        #     period=100
        # )
        # self.printer = BeforeEpoch()
        self.history = History(
            path=self.config.get("flow.premodel", "../data/models/"),
            add_keys=[
                "epoch_elapsed_time_in_seconds",
                "average_batch_elapsed_time_in_seconds",
                "learning_elapsed_time_in_seconds",

            ]
        )
        self._timers = Timers()

    def get_optimizer(self):
        if self.optimizer is None:
            # learning rate
            lr = tf.Variable(
                self.init_lr, 
                trainable=False, 
                # aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                # synchronization=tf.VariableSynchronization.NONE
            )
            self.init_lr = lr
            # Implements linear decay of the learning rate.
            # using compat.v1 module for compatibilty with tensorflow 1.x (old versions of tensorflow). 
            lr = LrScheduler(
                initial_learning_rate=lr, 
                decay_steps=self.per_epoch_steps * int(self.config.get("flow.n_epochs")), 
                warmup_steps=self.warmup_steps, 
                end_learning_rate=0.0, 
                power=1.0, 
                cycle=False,
            )

            # optimizer
            # optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            optimizer = tf.keras.optimizers.Adam(lr)
            self.optimizer = optimizer
        return self.optimizer

    def fit(
            self, train_dataset: Dataset, valid_dataset: Dataset=None,
            resume=False
    ):
        """
        Model fit function.
        :param train_dataset: training dataset partition.
        :param valid_dataset: valid dataset partition.
        :param resume: true if it is strating from an already run fitting state.
        """
        if not self.built:
            self.build()

        learner = Learner(
            model=self,
            resume=resume
        )
        # TODO prepare inputs: it could be possible to pass the inputs as parameters.
        learner.fit(
            train_dataset,
            valid_dataset
        )

    def load(self, path):
        """
        Loads a saved model.
        :param path: the saved model path.
        """
        if not self.built:
            self.build()

        self._checkpointer.load(path)
