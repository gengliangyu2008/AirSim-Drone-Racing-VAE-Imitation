import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape, ReLU
from tensorflow.keras.applications import DenseNet121
from datetime import datetime

# tf.config.experimental_run_functions_eagerly(True)
# with tf.device("/gpu:0"):

class Dronet(Model):
    def __init__(self, num_outputs, include_top=True):
        super(Dronet, self).__init__()
        self.include_top = include_top
        self.create_model(num_outputs)

    def call(self, img):
        # Input
        # print("==============img.shape:", img.shape)
        '''
        print('{} | img.shape: {},'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            img.shape))
        '''
        model_d = self.denseNet121(img)
        '''
        model_d = tf.keras.layers.Activation('relu')(model_d)
        x = tf.keras.layers.Flatten()(x7)
        model_d = self.dense0(model_d)
        model_d = self.dense1(model_d)
        model_d = self.dense2(model_d)
        '''

        return model_d

    def create_model(self, num_outputs):
        print('[Dronet] Starting dronet')

        #  input_shape=(224, 224, 3),
        self.denseNet121 = DenseNet121(include_top=self.include_top, weights=None, classes=num_outputs)

        print('[Dronet] Done with dronet')