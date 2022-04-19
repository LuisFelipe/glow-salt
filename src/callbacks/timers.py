# -*- coding: utf-8 -*-
"""
module early_stop.py
-----------------------
Calculates elapsed time during training procedure.
"""
import numpy as np
import tensorflow as tf
import time
from flow.callbacks import on_batch_begin, on_batch_end, on_epoch_begin, on_epoch_end, on_train_begin, \
    on_train_end, on_validate_begin, on_validate_end


class Timers(object):
    """
    Measure elapsed time for each learning step.
    """

    def __init__(self):
        """
        callback initialization.
        """
        self.learning_start_time = None
        self.learning_end_time = None
        self.current_batch_start_time = None
        self.current_batch_end_time = None
        self.batch_time_accumulator = list()
        self.current_epoch_start_time = None
        self.current_epoch_end_time = None
        self.current_validation_start_time = None
        self.current_validation_end_time = None

        on_train_begin.connect(self.on_train_begin, weak=False)
        on_train_end.connect(self.on_train_end, weak=False)
        on_epoch_begin.connect(self.on_epoch_begin, weak=False)
        on_epoch_end.connect(self.on_epoch_end, weak=False)
        on_train_begin.connect(self.on_train_begin, weak=False)
        on_train_end.connect(self.on_train_end, weak=False)
        on_batch_begin.connect(self.on_batch_begin, weak=False)
        on_batch_end.connect(self.on_batch_end, weak=False)
        on_validate_begin.connect(self.on_validate_begin, weak=False)
        on_validate_end.connect(self.on_validate_end, weak=False)

    def on_train_begin(self, sender):
        self.learning_start_time = time.time()

    def on_train_end(self, sender):
        self.learning_end_time = time.time()
        current_state = sender.current_state
        current_state["learning_elapsed_time_in_seconds"] = round(self.learning_end_time - self.learning_start_time)
        print(
            "\n\nLearning elapsed time {} minutes and {} seconds."
                .format(
                current_state["learning_elapsed_time_in_seconds"] // 60,
                current_state["learning_elapsed_time_in_seconds"] % 60
            )
        )

    def on_epoch_begin(self, sender):
        self.batch_time_accumulator.clear()
        self.current_epoch_start_time = time.time()

    def on_epoch_end(self, sender):
        self.current_epoch_end_time = time.time()
        current_state = sender.current_state
        current_state["epoch_elapsed_time_in_seconds"] = round(self.current_epoch_end_time - self.current_epoch_start_time)
        current_state["average_batch_elapsed_time_in_seconds"] = np.mean(self.batch_time_accumulator)

    def on_batch_begin(self, sender):
        self.current_batch_start_time = time.time()

    def on_batch_end(self, sender):
        self.current_batch_end_time = time.time()
        self.batch_time_accumulator.append(round(self.current_batch_end_time - self.current_batch_start_time))

    def on_validate_begin(self, sender):
        self.current_validation_start_time = time.time()

    def on_validate_end(self, sender):
        if self.current_validation_start_time is not None:
            self.current_validation_end_time = time.time()
            current_state = sender.current_state
            current_state["validation_elapsed_time_in_seconds"] = round(
                self.current_validation_end_time - self.current_validation_start_time
            )
