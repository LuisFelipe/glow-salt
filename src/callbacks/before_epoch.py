# -*- coding: utf-8 -*-
"""
module early_stop.py
-----------------------
Early Stop training Callback.
"""
import os
import numpy as np
import tensorflow as tf
from flow.callbacks import on_batch_begin, on_batch_end, on_epoch_begin, on_epoch_end, on_train_begin, \
    on_train_end, on_validate_begin, on_validate_end
from flow.callbacks import ModeEnum
from flow.config import Config
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.contrib.framework import arg_scope


class BeforeEpoch(object):
    """
    Stop training when a monitored quantity has stopped improving.
    """

    def __init__(self):
        """
        callback initialization.
        """
        self.config = Config()
        self.batch_count = 0
        on_epoch_end.connect(self.on_epoch_end, weak=False)
        # on_train_begin.connect(self.on_train_begin, weak=False)
        # on_batch_end.connect(self.on_batch_end, weak=False)

    def on_epoch_end(self, sender):
        # current = logs.get(self.monitor)
        should_print = self.config.get("flow.sample_and_print", "false").lower() == "true"
        if should_print:
            epoch = sender.current_state["current_epoch"]
            if epoch % 2 == 0:
                self._sample_and_print(sender)

    def _sample_and_print(self, sender):
        epoch = sender.current_state["current_epoch"]
        base_path = "../data/imgs"
        # on_epoch_begin.send(sender)
        on_validate_begin.send(sender)
        model = sender.model

        if not os.path.exists(base_path):
            os.mkdir(base_path)
        path = base_path + "/epoch-" + str(epoch)
        if not os.path.exists(path):
            os.mkdir(path)

        valid_ds = model._valid_dataset
        outs = model.predict(valid_ds, {"x": model.outputs.x, "input": model.inputs.y, "eps": model.outputs.eps,
                                        "z": model.outputs.z})
        count = 0
        for out in outs:
            for x, y, eps, z in zip(out["x"], out["input"], out["eps"], out["z"]):
                img = Image.fromarray(x)
                img = img.convert(mode="L")
                img.save(fp=path + "/{}-lab{}.jpg".format(count, y))
                img.close()
                img = Image.fromarray(x, mode='RGB')
                img.save(fp=path + "/{}RGB-lab{}.jpg".format(count, y))
                img.close()
                count += 1
                if count >= 5:
                    break
            if count >= 5:
                break
        model._valid_dataset = valid_ds

    def on_batch_end(self, sender):
        self.batch_count += 1
        epoch = sender.current_state["current_epoch"]
        base_path = "../data/imgs"
        if self.batch_count % 1000 == 0:
            on_epoch_begin.send(sender)
            model = sender.model

            if not os.path.exists(base_path):
                os.mkdir(base_path)
            path = base_path+"/epoch-"+str(epoch)+"-itr{}".format(self.batch_count)
            if not os.path.exists(path):
                os.mkdir(path)

            valid_ds = model._valid_dataset
            outs = model.predict(valid_ds, {"x": model.outputs.x, "input": model.inputs.y})
            count = 0
            for out in outs:
                for x, y in zip(out["x"], out["input"]):
                    img = Image.fromarray(x)
                    img = img.convert(mode="L")
                    img.save(fp=path+"/{}-lab{}.jpg".format(count, y))
                    img.close()
                    img = Image.fromarray(x, mode='RGB')
                    img.save(fp=path+"/{}RGB-lab{}.jpg".format(count, y))
                    img.close()
                    count += 1
                    if count >= 5:
                        break
                if count >= 5:
                    break

            model._valid_dataset = valid_ds

    def on_train_begin(self, sender):
        should_data_dependt_init = self.config.get("model.data_dependent_init", "true") == "true"
        is_distributed = self.config.get("flow.distributed", "false").lower() == "true"
        if should_data_dependt_init:
            if is_distributed:
                strategy = tf.distribute.get_strategy()
                with strategy.scope():
                    on_epoch_begin.send(sender)
                    sess = tf.get_default_session()
                    init_ops = tf.get_collection("actnorm")
                    for op in init_ops:
                        try:
                            sess.run(op)
                        except tf.errors.OutOfRangeError:
                            print(">>>>out_of_range_error>>>", op)
                            on_epoch_begin.send(sender)
                            sess.run(op)
            else:
                on_epoch_begin.send(sender)
                sess = tf.get_default_session()
                init_ops = tf.get_collection("actnorm")
                # sess.run(tf.group(init_ops))
                for op in init_ops:
                    try:
                        sess.run(op)
                    except tf.errors.OutOfRangeError:
                        print(">>>>out_of_range_error>>>", op)
                        on_epoch_begin.send(sender)
                        sess.run(op)

                evaluations = sender.model.evaluate(sender.model._train_dataset)
                print("*****init_train metrics****", evaluations)
