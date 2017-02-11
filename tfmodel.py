from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class TFModel(object):

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def compute_gradients(self):
        raise NotImplementedError

    def model_update(self, gradients):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, list_weights):
        raise NotImplementedError
