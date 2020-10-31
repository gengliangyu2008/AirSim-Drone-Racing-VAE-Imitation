# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K

# Creating Densenet 121
'''
def densenet(input_shape, n_classes, filters=32):
    
    #batchnorm + relu + conv
    def bn_rl_conv(x, filters, kernel=1, strides=1):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
        return x
    
    def dense_block(x, repetition):
        for _ in range(repetition):
            y = bn_rl_conv(x, 4*filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y,x])
        return x
    
    def transition_layer(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x
    
    input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    for repetition in [6, 12, 24, 16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)
        
    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation='softmax')(x)
    
    model = Model(input, output)
    return model

input_shape = 224, 224, 3
n_classes = 3

model = densenet(input_shape, n_classes)
model.summary()    
'''

class Dronet(Model):
    def __init__(self, input_shape, num_outputs, filters=32):
        super(Dronet, self).__init__()
        self.input_shapeA = input_shape
        self.n_classes = num_outputs
        self.filters = filters
        self.customized_DenseNet121 = None
        self.create_model()

    def call(self, img):
        # Input
        model_d = self.customized_DenseNet121(img)

        return model_d

    def create_model(self):
        print('[Dronet] Starting customized_DenseNet121')

        input = Input(self.input_shapeA)
        x = Conv2D(64, 7, strides=2, padding='same')(input)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        for repetition in [6, 12, 24, 16]:
            d = self.dense_block(x, repetition)
            x = self.transition_layer(d)

        x = GlobalAveragePooling2D()(d)
        output = Dense(self.n_classes, activation='softmax')(x)

        self.customized_DenseNet121 = Model(input, output)

        print('[Dronet] Done with customized_DenseNet121')

    # batchnorm + relu + conv
    def bn_rl_conv(self, x, filters, kernel=1, strides=1):
        bn_rl_conv = BatchNormalization()(x)
        bn_rl_conv = ReLU()(bn_rl_conv)
        bn_rl_conv = Conv2D(filters, kernel, strides=strides, padding='same')(bn_rl_conv)
        return bn_rl_conv

    def dense_block(self, x, repetition):
        for _ in range(repetition):
            y = self.bn_rl_conv(x, 4 * self.filters)
            y = self.bn_rl_conv(y, self.filters, 3)
            x = concatenate([y, x])
        return x

    def transition_layer(self, x):
        x = self.bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x