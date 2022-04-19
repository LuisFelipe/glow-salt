# -*- coding: utf-8 -*-
"""
Module collection.py
----------------------
A utility class to easily store and recover tensor from anywhere in the source code.
Defines a globally accessible named collection of tensors.
When used in the eager execution mode of tensorflow it must be cleaned after each iteration step.
"""

# a marker that identifies that no value was passed to an optional named parameter
# this marker is used for named parameters that can the default value can not be None
_marker = object()


class Collection(dict):
    """A Globally accessible named collection of tensors.

    A Globally accessible named collection of tensors.
    When used in the eager execution mode of tensorflow, 
    it must be cleaned after each iteration step.
    """
    __registry__ = dict()

    def __new__(cls, name="Default", defaults=None):
        """
        Class constructor.
        :param name: instance name id.
        :param defaults: dafault values dict or None.
        :return: an instance of the collection object.
        """
        if name not in cls.__registry__:
            cls.__registry__[name] = dict.__new__(cls)
        return cls.__registry__[name]

    def __init__(self, name="Default", defaults=None):
        if not hasattr(self, "name"):
            dict.__init__(self, defaults or {})
            self.name = name

    def __repr__(self):
        return '<{} name: {} {}>'.format(
            self.__class__.__name__, 
            self.name, 
            dict.__repr__(self)
        )

    def __getitem__(self, name):
        name = name.upper()
        tensor = self.get(name)
        if tensor is None and tensor not in self:
            raise KeyError(name)
        return tensor

    def __setitem__(self, key, value):
        self.set(key, value)

    def get(self, name, default=_marker):
        """
        Gets a tensor from the collection within a name or a default value.
        If no default value is given and there ir no tensor in the collection 
        within the given name, then it will return **None**.

        :param name: a tensor name.
        :param default: a default value to be returned when the name is not found.
        :return: a tf.Tensor from the collection or a default value.
                If no default value is given and there ir no tensor in the collection 
                within the given name, then it will return None. 
        """
        name = name.upper()
        tensor = dict.get(self, name, default)

        if tensor is _marker:
            return None
        return tensor

    def set(self, name, tensor):
        """
        Sets a name/tensor to the collection.
        :param name: the tensor name.
        :param tensor: a tf.Tensor.
        """
        name = name.upper()
        if name in self:
            raise ValueError(
                "A tensor named '{}' already exists in the collection." 
                "\nCollection name: '{}'".format(name, self.name)
            )
        dict.__setitem__(self, name, tensor)

    @classmethod
    def clear_collection(cls, name):
        if name not in cls.__registry__:
            raise KeyError(name)
        collection = cls.__registry__[name]

        for key in list(collection.keys()):
            t = collection.pop(key)
            del t

        collection.clear()
        cls.__registry__.pop(name)

    @classmethod
    def clear_all_collections(cls):
        keys = list(cls.__registry__.keys())
        for name in keys:
            cls.clear_collection(name)


def add_to_collection(collection_name, tensor, tensor_name=None):
    collection = Collection(name=collection_name)

    if tensor_name is None:
        tensor_name = tensor.name

    collection.set(tensor_name, tensor)


def get_collection(collection_name):
    collection = Collection(name=collection_name)
    return collection


def get_tensor_from_collection(collection_name, tensor_name, default=_marker):
    collection = Collection(name=collection_name)

    if default is _marker:
        return collection[tensor_name]

    return collection.get(tensor_name, default=default)


def clear_all():
    Collection.clear_all_collections()


def clear_collection(collection_name):
    Collection.clear_collection(collection_name)
