import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape, ReLU
from tensorflow.keras.applications import DenseNet121

class Dronet(Model):
    def __init__(self, num_outputs, include_top=True):
        super(Dronet, self).__init__()
        self.include_top = include_top
        self.create_model(num_outputs)
        
    @tf.function
    def call(self, img):
        # Input
        x = DenseNet121(include_top=self.include_top, weights=None, classes = 10) (img)
        if self.include_top:
            x = tf.keras.layers.Activation('relu')(x)
            # x = tf.keras.layers.Dropout(0.5)(x)
            x = self.dense0(x)
            x = self.dense1(x)
            gate_pose = self.dense2(x)
            # phi_rel = self.dense_phi_rel(x)
            # gate_pose = tf.concat([gate_pose, phi_rel], 1)
            return gate_pose
        else:
            return x
    @tf.function
    def create_model(self, num_outputs):
        print('[Dronet] Starting dronet')

        self.dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=num_outputs, activation='linear')

        print('[Dronet] Done with dronet')