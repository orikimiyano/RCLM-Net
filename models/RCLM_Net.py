import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from layers.rr_conv import RR_CONV
from models.loss import *

IMAGE_SIZE = 256
filter = 24

activation_value = 'LeakyReLU'
batch_norm_value = False


# LRCNet 1版

# filters,kernel_size,strides

def net(pretrained_weights=None, input_size=(IMAGE_SIZE, IMAGE_SIZE, 1), num_class=20):
    input_1 = Input(input_size)
    input_2 = Input(input_size)
    input_3 = Input(input_size)

    #####-----inputs_plug------#####
    conv_input_1 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(input_1)
    conv_input_1 = LeakyReLU(alpha=0.3)(conv_input_1)
    conv_input_2 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(input_2)
    conv_input_2 = LeakyReLU(alpha=0.3)(conv_input_2)
    conv_input_3 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(input_3)
    conv_input_3 = LeakyReLU(alpha=0.3)(conv_input_3)

    merge_ip1 = concatenate([conv_input_1, conv_input_2], axis=3)
    merge_ip2 = concatenate([conv_input_2, conv_input_3], axis=3)
    merge_ip3 = concatenate([conv_input_1, conv_input_3], axis=3)

    conv_input_4 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(merge_ip1)
    conv_input_4 = LeakyReLU(alpha=0.3)(conv_input_4)
    conv_input_5 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(merge_ip2)
    conv_input_5 = LeakyReLU(alpha=0.3)(conv_input_5)
    conv_input_6 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(merge_ip3)
    conv_input_6 = LeakyReLU(alpha=0.3)(conv_input_6)

    fused_1 = Concatenate()([conv_input_4, conv_input_5, conv_input_6])
    #####-----inputs_plug------#####

    #####-----Hierarchical-Block-1------#####
    # ConvBlock_1_64
    conv1 = RR_CONV(fused_1, filter, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr1')
    merge1 = concatenate([conv1, fused_1], axis=3)
    conv1 = LeakyReLU(alpha=0.3)(merge1)

    # ConvBlock_2_64
    conv2 = RR_CONV(conv1, filter, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr2')
    merge2 = concatenate([conv2, conv1], axis=3)
    conv2 = LeakyReLU(alpha=0.3)(merge2)

    # AggregationBlock_3_64
    merge_unit1 = concatenate([conv1, conv2], axis=3)
    conv_root3 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(merge_unit1)
    Aggre3 = LeakyReLU(alpha=0.3)(conv_root3)
    #####-----Hierarchical-Block-1------#####

    pool3 = MaxPool2D(pool_size=(2, 2))(Aggre3)

    #####-----Hierarchical-Block-2------#####
    # ConvBlock_4_128
    conv4 = RR_CONV(pool3, filter * 2, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr4')
    merge4 = concatenate([conv4, pool3], axis=3)
    conv4 = LeakyReLU(alpha=0.3)(merge4)

    # ConvBlock_5_128
    conv5 = RR_CONV(conv4, filter * 2, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr5')
    merge5 = concatenate([conv4, conv5], axis=3)
    conv5 = LeakyReLU(alpha=0.3)(merge5)

    # AggregationBlock_6_128
    merge_unit2 = concatenate([conv4, conv5], axis=3)
    conv_root6 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(merge_unit2)

    mergeA1_A2 = concatenate([conv_root6, pool3], axis=3)
    conv_root6 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(mergeA1_A2)

    Aggre6 = LeakyReLU(alpha=0.3)(conv_root6)
    #####-----Hierarchical-Block-2------#####

    pool6 = MaxPool2D(pool_size=(2, 2))(Aggre6)

    #####-----Hierarchical-Block-3------#####
    # ConvBlock_7_256
    conv7 = RR_CONV(pool6, filter * 4, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr7')
    merge7 = concatenate([conv7, pool6], axis=3)
    conv7 = LeakyReLU(alpha=0.3)(merge7)

    # ConvBlock_8_256
    conv8 = RR_CONV(conv7, filter * 4, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr8')
    merge8 = concatenate([conv7, conv8], axis=3)
    conv8 = LeakyReLU(alpha=0.3)(merge8)

    # AggregationBlock_9_256
    merge_unit3 = concatenate([conv7, conv8], axis=3)
    conv_root9 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(merge_unit3)

    mergeA2_A3 = concatenate([conv_root9, pool6], axis=3)
    conv_root9 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(mergeA2_A3)

    Aggre9 = LeakyReLU(alpha=0.3)(conv_root9)
    #####-----Hierarchical-Block-3------#####

    #####-----Center-Block-1------#####
    # ConvC1_Block_16
    merge_unit_C1 = concatenate([pool3, Aggre6], axis=3)
    convC1 = RR_CONV(merge_unit_C1, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr_c11')

    AggreC1 = RR_CONV(convC1, filter * 2, stack_num=2, recur_num=2,
                      activation=activation_value, batch_norm=batch_norm_value, name='rr_c12')

    # ConvC3_Block_18
    merge_unit_C3_1 = concatenate([pool3, AggreC1], axis=3)
    AggreC3_1 = RR_CONV(merge_unit_C3_1, filter * 4, stack_num=2, recur_num=2,
                        activation=activation_value, batch_norm=batch_norm_value, name='rr_c31')
    AggreC3_1 = MaxPool2D(pool_size=(2, 2))(AggreC3_1)

    merge_unit_C3_2 = concatenate([pool6, Aggre9], axis=3)
    convC3 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(merge_unit_C3_2)
    convC3 = BatchNormalization()(convC3)
    AggreC3_2 = LeakyReLU(alpha=0.3)(convC3)
    AggreC3_2 = RR_CONV(AggreC3_2, filter * 4, stack_num=2, recur_num=2,
                        activation=activation_value, batch_norm=batch_norm_value, name='rr_c32')

    merge_unit_C3_3 = concatenate([AggreC3_1, AggreC3_2], axis=3)
    AggreC3_3 = RR_CONV(merge_unit_C3_3, filter * 4, stack_num=2, recur_num=2,
                        activation=activation_value, batch_norm=batch_norm_value, name='rr_c33')

    # ConvC2_Block_17
    convC2 = RR_CONV(AggreC3_3, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr_c21')
    AggreC2 = RR_CONV(convC2, filter * 2, stack_num=2, recur_num=2,
                      activation=activation_value, batch_norm=batch_norm_value, name='rr_c22')
    #####-----Center-Block-1------#####

    up9 = UpSampling2D(size=(2, 2))(Aggre9)

    #####-----Hierarchical-Block-4------#####
    # ConvBlock_10_128
    conv10 = RR_CONV(up9, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr10')

    merge10 = concatenate([conv10, up9], axis=3)
    conv10 = LeakyReLU(alpha=0.3)(merge10)

    # ConvBlock_11_128
    conv11 = RR_CONV(conv10, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr11')
    merge11 = concatenate([conv10, conv11], axis=3)
    conv11 = LeakyReLU(alpha=0.3)(merge11)

    # AggregationBlock_12_128
    merge_unit4 = concatenate([conv10, conv11], axis=3)
    conv_root12 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(merge_unit4)
    Aggre12 = LeakyReLU(alpha=0.3)(conv_root12)

    # con_up12 = UpSampling2D(size=(2, 2))(Aggre9)
    skip12 = concatenate([Aggre12, up9], axis=3)
    conv_skip12 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(skip12)
    conv_skip12 = LeakyReLU(alpha=0.3)(conv_skip12)

    skip13 = concatenate([AggreC3_3, AggreC2], axis=3)
    conv_skip13 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(skip13)
    conv_skip13 = LeakyReLU(alpha=0.3)(conv_skip13)
    conv_skip13 = UpSampling2D(size=(2, 2))(conv_skip13)

    skip14 = concatenate([conv_skip12, conv_skip13], axis=3)
    conv_skip14 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(skip14)
    conv_skip14 = LeakyReLU(alpha=0.3)(conv_skip14)
    #####-----Hierarchical-Block-4------#####

    up12 = UpSampling2D(size=(2, 2))(conv_skip14)

    #####-----Hierarchical-Block-5------#####
    # ConvBlock_13_64
    conv13 = RR_CONV(up12, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr13')

    merge13 = concatenate([conv13, up12], axis=3)
    conv13 = LeakyReLU(alpha=0.3)(merge13)

    # ConvBlock_14_64
    conv14 = RR_CONV(conv13, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr14')
    merge14 = concatenate([conv13, conv14], axis=3)
    conv14 = LeakyReLU(alpha=0.3)(merge14)

    # AggregationBlock_15_64
    merge_unit5 = concatenate([conv13, conv14], axis=3)
    conv_root15 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(merge_unit5)
    Aggre15 = LeakyReLU(alpha=0.3)(conv_root15)

    # con_up15 = UpSampling2D(size=(4, 4))(conv_skip9)
    skip15 = concatenate([Aggre15, up12], axis=3)
    conv_skip15 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(skip15)
    conv_skip15 = LeakyReLU(alpha=0.3)(conv_skip15)

    skip16 = concatenate([AggreC3_3, AggreC2], axis=3)
    skip16 = UpSampling2D(size=(4, 4))(skip16)
    conv_skip16 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(skip16)
    conv_skip16 = LeakyReLU(alpha=0.3)(conv_skip16)

    skip17 = concatenate([conv_skip15, conv_skip16], axis=3)
    conv_skip17 = RR_CONV(skip17, filter, stack_num=2, recur_num=2,
                          activation=activation_value, batch_norm=batch_norm_value, name='conv_out')
    #####-----Hierarchical-Block-5------#####

    conv_out = Conv2D(num_class, 1, activation='softmax')(conv_skip17)
    # loss_function = 'categorical_crossentropy'

    model = Model(inputs=[input_1, input_2, input_3], outputs=conv_out)

    model.compile(optimizer=adam_v2.Adam(lr=1e-5), loss=[CE_plus_TV], metrics=['accuracy'])
    # model.summary()

    return model
