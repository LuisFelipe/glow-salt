# -*- coding: utf-8 -*-
"""
module lr_scheduler.py
--------------------
A custom learning rate scheduler used during our optimization procedure.
It implements an polinomial decay with linear warmup steps scheduler 
"""
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import numpy as np
from flow.config import Config
import collection


class LrScheduler(LearningRateSchedule):
    """A LearningRateSchedule that uses a polynomial decay with linear warmup schedule."""

    def __init__(
        self, initial_learning_rate,
        decay_steps,
        warmup_steps,
        end_learning_rate=0.0001,
        power=1.0,
        cycle=False,
        name=None
    ):
        """ Applies a warmup and a polynomial decay to the learning rate.

        It is commonly observed that a monotonically decreasing learning rate, whose
        degree of change is carefully chosen, results in a better performing model.
        This schedule applies a polynomial decay function to an optimizer step,
        given a provided `initial_learning_rate`, to reach an `end_learning_rate`
        in the given `decay_steps`.
        It requires a `step` value to compute the decayed learning rate. You
        can just pass a TensorFlow variable that you increment at each training
        step.
        The schedule is a 1-arg callable that produces a decayed learning rate
        when passed the current optimizer step. This can be useful for changing the
        learning rate value across different invocations of optimizer functions.
        It is computed as:
        ```python
        def decayed_learning_rate(step):
          step = min(step, decay_steps)
          return ((initial_learning_rate - end_learning_rate) *
                  (1 - step / decay_steps) ^ (power)
                 ) + end_learning_rate
        ```
        If `cycle` is True then a multiple of `decay_steps` is used, the first one
        that is bigger than `step`.
        ```python
        def decayed_learning_rate(step):
          decay_steps = decay_steps * ceil(step / decay_steps)
          return ((initial_learning_rate - end_learning_rate) *
                  (1 - step / decay_steps) ^ (power)
                 ) + end_learning_rate
        ```
        You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
        as the learning rate.
        Example: Fit a model while decaying from 0.1 to 0.01 in 10000 steps using
        sqrt (i.e. power=0.5):
        ```python
        ...
        starter_learning_rate = 0.1
        end_learning_rate = 0.01
        decay_steps = 10000
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            starter_learning_rate,
            decay_steps,
            end_learning_rate,
            power=0.5)
        model.compile(optimizer=tf.keras.optimizers.SGD(
                          learning_rate=learning_rate_fn),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(data, labels, epochs=5)
        ```
        The learning rate schedule is also serializable and deserializable using
        `tf.keras.optimizers.schedules.serialize` and
        `tf.keras.optimizers.schedules.deserialize`.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
          warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.
          end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The minimal end learning rate.
          power: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The power of the polynomial. Defaults to linear, 1.0.
          cycle: A boolean, whether or not it should cycle beyond decay_steps.
          name: String.  Optional name of the operation. Defaults to
            'PolynomialDecay'.
        Returns:
          A 1-arg callable learning rate schedule that takes the current optimizer
          step and outputs the decayed learning rate, a scalar `Tensor` of the same
          type as `initial_learning_rate`.
        """
        super(LrScheduler, self).__init__()

        self.polynomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate, 
            decay_steps=decay_steps, 
            end_learning_rate=end_learning_rate, 
            power=power, 
            cycle=cycle, 
            name=name
        )
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.name = name
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        with tf.name_scope(self.name or "LrScheduler"):
            lr = self.polynomial_decay(step)

            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, 
                name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype

            global_step = tf.cast(step, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)

            # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            warmup_percent = global_step / warmup_steps
            warmup_learning_rate = initial_learning_rate * warmup_percent

            is_warmup = tf.cast(global_step < warmup_steps, tf.float32)
            lr = (1.0 - is_warmup) * lr + (is_warmup * warmup_learning_rate)
            return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "cycle": self.cycle,
            "name": self.name,
            "warmup_steps": self.warmup_steps
        }

