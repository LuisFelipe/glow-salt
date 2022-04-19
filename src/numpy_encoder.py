# -*- coding: utf-8 -*-
"""
module numpy_encoder.py
--------------------
A Json encoder that deals with 'numpy.ndarray' serialization.
"""
import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    """
    A JSONEncoder sub-class that is able to serialize numpy arrays.
    To use this encoder instead of the default json encoder you must call json.dump or json.dumps
    passing this class to the 'cls' optional named parameter as the follows.

        'json.dumps(array, cls=NumpyEncoder)'
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.core.floating):
            return float(o)
        if isinstance(o, np.core.signedinteger):
            return int(o)

        return json.JSONEncoder.default(self, o)
