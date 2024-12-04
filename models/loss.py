from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf
from keras.losses import categorical_crossentropy
import math

smooth = 1.

# ——————————
# Cross_entropy_loss
# ——————————
def Cross_entropy_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    crossEntropyLoss = -y_true * K.log(y_pred)

    return tf.reduce_sum(crossEntropyLoss, -1)

# ——————————
# tversky_loss
# ——————————
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


# ——————————
# CE_plus_TV
# ——————————

def CE_plus_TV(y_true, y_pred):
    a = 0.7
    return a*tversky_loss(y_true, y_pred) + (1-a)*Cross_entropy_loss(y_true, y_pred)
