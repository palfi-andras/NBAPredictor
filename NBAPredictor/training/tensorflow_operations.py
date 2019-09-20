from __future__ import absolute_import, division, print_function, unicode_literals
from league import League
from read_game import ReadGame
import numpy as np
import tensorflow as tf


class TensorflowOperations:

    def __init__(self):
        pass

    def get_tensorflow_dataset(self, parsed_game: ReadGame):
        return tf.data.Dataset.from_tensor_slices((parsed_game.inputs, parsed_game.labels))