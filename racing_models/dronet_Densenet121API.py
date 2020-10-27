import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape, ReLU
from tensorflow.keras.applications import DenseNet121

# tf.config.experimental_run_functions_eagerly(True)
# with tf.device("/gpu:0"):

class Dronet(Model):
    def __init__(self, num_outputs, include_top=True):
        super(Dronet, self).__init__()
        self.include_top = include_top
        self.create_model(num_outputs)

    def call(self, img):
        # Input
        # x = DenseNet121(include_top=self.include_top, weights=None, classes = 10) (img)
        # model_d = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        model_d = self.denseNet121(img)
        '''
        model_d = tf.keras.layers.Activation('relu')(model_d)
        x = tf.keras.layers.Flatten()(x7)
        model_d = self.dense0(model_d)
        model_d = self.dense1(model_d)
        model_d = self.dense2(model_d)
        '''

        return model_d

        '''
        model_d = tf.keras.layers.Activation('relu')(model_d)
        model_d = self.dense0(model_d)
        model_d = self.dense1(model_d)
        gate_pose = self.dense2(model_d)
        return gate_pose
        '''

    def create_model(self, num_outputs):
        print('[Dronet] Starting dronet')

        # self.dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        # self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        # self.dense2 = tf.keras.layers.Dense(units=num_outputs, activation='linear')

        self.denseNet121 = DenseNet121(include_top=self.include_top, weights=None, classes=20)
        '''
        self.dense0 = DenseNet121(include_top=self.include_top, weights=None, classes=num_outputs)
        self.dense1 = DenseNet121(include_top=self.include_top, weights=None, classes=num_outputs)
        self.dense2 = DenseNet121(include_top=self.include_top, weights=None, classes=num_outputs)
        '''
        print('[Dronet] Done with dronet')