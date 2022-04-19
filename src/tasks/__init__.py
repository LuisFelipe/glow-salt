# -*- coding: utf-8 -*-
"""
package tasks
--------------------
A set of scripts to run the model tasks.
"""
from invoke import Collection
from . import run, ds

ns = Collection(run, ds)
