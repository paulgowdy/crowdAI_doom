import tensorflow as tf
import numpy as np

# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels)
        return tf.random_uniform(shape, minval=-d, maxval=d)
    return _initializer

def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
        return tf.random_uniform(shape, minval=-d, maxval=d)
    return _initializer

class UnrealModel(object):

    def __init__(self,action_size,thread_index, use_pixel_change,use_value_replay,use_reward_prediction,pixel_change_lambda,entropy_beta,device,for_display=False):

        self._device = device
        self._action_size = action_size
        self._thread_index = thread_index
        self._use_pixel_change = use_pixel_change
        self._use_value_replay = use_value_replay
        self._use_reward_prediction = use_reward_prediction
		self._pixel_change_lambda = pixel_change_lambda
		self._entropy_beta = entropy_beta
