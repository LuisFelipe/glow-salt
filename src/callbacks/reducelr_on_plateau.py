# -*- coding: utf-8 -*-
"""
module checkpointer.py
--------------------------------
Saves the current tensorflow graph state during learning procedure.
"""
import numpy as np
import tensorflow as tf
import tensorflow
import os
import tarfile
import json
from flow.callbacks import ModeEnum
from flow.callbacks import on_batch_begin, on_batch_end, on_epoch_begin, on_epoch_end, on_train_begin, \
    on_train_end, on_validate_begin, on_validate_end


class ReduceLROnPlateau(object):

    def __init__(
            self,
            model,
            factor,
            monitor='val_loss',
            verbose=0,
            mode: ModeEnum = ModeEnum.MIN,
            period=1,
    ):
        self.model = model
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.epochs_since_last_improve = 0
        self.mode = mode
        self.factor = factor
        if mode is ModeEnum.MIN:
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode is ModeEnum.MAX:
            self.monitor_op = np.greater
            self.best = -np.Inf

        on_epoch_end.connect(self.on_epoch_end, weak=False)

    def on_epoch_end(self, sender):
        epoch = sender.current_state["current_epoch"]
        self.epochs_since_last_improve += 1
        self._do(epoch, sender)

    def _do(self, epoch, sender):

        current = sender.current_state.get(self.monitor, None)

        if current is None:
            if self.verbose > 0:
                print('Can not evaluate metric. {} is not available. '
                            'skipping.'.format(self.monitor))
        else:
            if self.monitor_op(current, self.best):
                self.best = current
                self.epochs_since_last_improve = 0
            else:
                if self.epochs_since_last_improve >= self.period:
                    lr = self.model.optimizer.lr.initial_learning_rate
                    strategy = tf.distribute.get_strategy()
                    strategy.run(self.reduce_fn, args=(lr,))
                    # strategy.experimental_run_v2(self.reduce_fn, args=(lr,))

                    if self.verbose > 0:
                        print(
                            '{} epochs without improve. Reducing Learning Rate.'.format(
                                self.epochs_since_last_improve
                            )
                        )
    @tf.function
    def reduce_fn(self, lr):
        # lr = lr[0]
        return lr.assign(lr * self.factor)
