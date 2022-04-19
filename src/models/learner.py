# -*- coding: utf-8 -*-
"""
module learner.py
--------------------
A base class definition for the learner class.
The default learner defines steps for the learning procedure.
"""
import tensorflow as tf
import numpy as np
from flow.config import Config
from flow.callbacks import before_session_initialization, on_batch_begin, on_batch_end, on_train_begin, on_epoch_begin
from flow.callbacks import validate_sig, on_validate_begin, on_validate_end, on_epoch_end, on_train_end
from tqdm import tqdm
import collection
import utils


class Learner(object):
    """
    Defines the learning procedure for fitting models.
    """

    def __init__(self, model, resume=False):
        """
        Default learner initializer.
        :param model: reference to the model object
        :param outputs: a dictionary of outputs.
        :param optimizer: the optimizer to be used.
        :param loss_name: loss tensor name.
        """
        self.model = model
        self.optimizer = model.optimizer

        self.config = self.model.config
        self.current_state = dict()
        self._train_ds = None
        self._valid_ds = None
        # flag used to signaling when it should stop learning loop.
        self._stop_flag = False

        self.strategy = utils.get_distribution_strategy()
        self.resume = resume
        # if resume:
        #     p = self.config.get("flow.premodel")
        #     self.model.load(tf.train.latest_checkpoint(p))

    def _get_accumulators(self):
        """
        builds and returns a dictionary containing a key for each step output.
        The outputs of each learning step is stored on this dictionary.
        :param outputs: A dictionary containing the learning step outputs.
        """
        accumulators = dict()
        return accumulators

    def _add_to_accumulator(self, accumulator, name, value):
        if name not in accumulator:
            accumulator[name] = list()

        accumulator[name].append(
            value
        )

    def _aggregate_accumulators(self, accumulators: dict):
        """
        Averages all learning step outputs for one epoch.
        :param accumulators: epoch outputs values.
        """
        # on epoch end triggers
        for output_name, output_vals in accumulators.items():
            self.current_state[output_name] = np.mean(output_vals)

    def epoch_fit_loop(self):
        """
        Fit inner loop for one learning epoch.
        :param outputs: a dictionary of outputs.
        """
        tf.keras.backend.set_learning_phase(True)
        accumulators = self._get_accumulators()

        for inputs in self._train_ds.dataset:
            # on batch begin triggers
            on_batch_begin.send(self)
            # run session and and perform learning step.
            outputs, names = self.learn_step(inputs)
            
            # # accumulate outputs
            for t, name in zip(outputs, names):
                # name = t._name
                self._add_to_accumulator(
                    accumulators,
                    name.numpy().decode("utf-8"),
                    t.numpy()
                    # t
                )

            # on batch end triggers
            del inputs
            on_batch_end.send(self)

        return accumulators

    @tf.function
    def learn_step(self, inputs):
        """defines one leaning iteration step."""
        # per_replica_results = self.strategy.experimental_run_v2(
        per_replica_results = self.strategy.run(
            self._step_fn, args=inputs
        )
        per_replica_loss = per_replica_results[0]
        per_replica_metrics = per_replica_results[1:]
        loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, 
            per_replica_loss,
            axis=None
        )

        loss._name = "loss"
        m_names = ["bits_x", "bits_y", "acc"]
        names = ["loss"]
        metrics = []

        for per_replica_m, name in zip(per_replica_metrics, m_names):
            m = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, 
                per_replica_m,
                axis=None
            )
            m._name = name
            metrics.append(m)
            names.append(name)

        return [loss] + metrics, names

    # @tf.function
    def _step_fn(self, *inputs):
        # add inputs to inputs collection
        for t, name in zip(inputs, self._train_ds.output_names):
            collection.add_to_collection(
                collection_name="inputs", 
                tensor=t, 
                tensor_name=name
            )
        with tf.GradientTape() as t:
            outputs = self.model(inputs[0])
            loss, metrics = self.model.loss_and_metrics(outputs, inputs[1])

        grads = t.gradient(
            loss, 
            self.model.trainable_variables
        )
        # clip gradients by norm
        clip_norm = float(self.config.get("optimizer.clip_norm", 4))
        grads = [tf.clip_by_norm(grad, clip_norm) for grad in grads]
        # tf.print("Learning Rate:", self.optimizer.lr(self.optimizer.iterations))
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        # clear all collections
        collection.clear_all()
        return [loss] + metrics

    def fit(self, train_ds, valid_ds):
        """
        Learning fit function. Starts the fitting procedure.
        :return:
        """
        self._train_ds = train_ds
        
        self._stop_flag = False
        # main loop
        with self.strategy.scope():
            # on train begin triggers
            on_train_begin.send(self)
            inputs = next(iter(self._train_ds.dataset))
            
            if self.resume:
                outputs, names = self.learn_step(inputs)
                p = self.config.get("flow.checkpoint")
                self.model.load(tf.train.latest_checkpoint(p))
                restart = lambda: [var.assign(tf.zeros_like(var)) for var in self.optimizer.variables()]
                self.strategy.run(
                     tf.function(restart)
                )
            else:
                self.data_dependet_init(inputs)

            del inputs

            for epoch_i in range(int(self.model.config["FLOW.N_EPOCHS"])):
                self.current_state["current_epoch"] = epoch_i
                # on epoch begin trigger
                on_epoch_begin.send(self)

                # epoch begin trigger
                accumulators = self.epoch_fit_loop()
                self._aggregate_accumulators(accumulators)

                if valid_ds is not None:
                    on_validate_begin.send(self)
                    results = self.evaluate(valid_ds)
                    self.current_state.update(results)
                    on_validate_end.send(self)

                on_epoch_end.send(self)
                tf.print("Learning Rate ep-{}:".format(epoch_i), self.optimizer.lr(self.optimizer.iterations))

                # print(">>>>current_state>>>", self.current_state)
                print(
                    "\n Epoch '{i}'=> loss: {loss:0.5f}, ".format(
                        i=self.current_state["current_epoch"] + 1,
                        loss=self.current_state["loss"]
                    ), self.current_state
                )

                # checks when the stop train flag is set to true
                # and break the main training loop when it happens
                if self._stop_flag:
                    break

        # on train end triggers
        on_train_end.send(self)
        # remove ds reference from learner
        self._train_ds = None

    def evaluate(self, ds):
        tf.keras.backend.set_learning_phase(False)
        accumulators = self._get_accumulators()

        dataset_len = len(ds)
        batch_size = int(self.config.get("FLOW.BATCH_SIZE", 1))

        # progress bar instance, giving running feedback
        progress_bar = tqdm(
            desc="Calculating metrics over the dataset.",
            total=dataset_len
        )

        for inputs in ds.dataset:
            for t, name in zip(inputs, ds.output_names):
                collection.add_to_collection(
                    collection_name="inputs", 
                    tensor=t, 
                    tensor_name=name
                )

            # per_replica_results = self.strategy.experimental_run_v2(
            per_replica_results = self.strategy.run(
                self.eval_step, args=tuple(inputs)
            )

            per_replica_loss = per_replica_results[0]
            per_replica_metrics = per_replica_results[1:]

            loss = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, 
                per_replica_loss
            )

            metrics = [
                self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, 
                    per_replica_m
                )
                for per_replica_m in per_replica_metrics
            ]

            # update accumulators
            self._add_to_accumulator(
                accumulators,
                loss._name,
                loss
            )

            for t in metrics:
                name = t._name
                self._add_to_accumulator(
                    accumulators,
                    name,
                    t
                )

            progress_bar.update(n=batch_size)

        progress_bar.close()

        # Aggregate result metrics
        results = dict()
        for output_name, output_vals in tqdm(
            accumulators.items(),
            desc="Aggregating evaluation metrics."
        ):
            size = len(output_vals)
            results["valid_" + output_name] = np.sum(output_vals) / (size * batch_size)

        return results

    @tf.function
    def eval_step(self, *inputs):
        outputs = self.model(inputs[0])
        loss, metrics = self.model.loss_and_metrics(outputs, inputs[1])

    @tf.function
    def data_dependet_init(self, inputs):
        # per_replica_results = self.strategy.experimental_run_v2(
        per_replica_results = self.strategy.run(
            self.model, args=(inputs[0],)
        )
