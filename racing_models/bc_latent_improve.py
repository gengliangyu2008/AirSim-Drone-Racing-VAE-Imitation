import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape, ReLU
from tensorflow.python.keras.layers import Dropout


class BcLatent(Model):
    def __init__(self):
        super(BcLatent, self).__init__()
        self.create_model()

    def call(self, z):
        return self.network(z)

    def create_model(self):
        print('[BcLatent-Improve] Starting create_model')

        # activation='sigmoid'
        dense0 = tf.keras.layers.Dense(units=256, activation='relu')
        drop_out0 = Dropout(0.2)
        #dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        drop_out2 = Dropout(0.15)
        #dense3 = tf.keras.layers.Dense(units=32, activation='relu')
        dense4 = tf.keras.layers.Dense(units=16, activation='relu')
        drop_out4 = Dropout(0.1)
        #dense5 = tf.keras.layers.Dense(units=8, activation='relu')
        dense6 = tf.keras.layers.Dense(units=4, activation='linear')
        drop_out6 = Dropout(0.05)

        self.network = tf.keras.Sequential([
            dense0,
            drop_out0,
            #dense1,
            dense2,
            drop_out2,
            #dense3,
            dense4,
            drop_out4,
            #dense5,
            dense6,
            drop_out6
        ], name='bc_dense')

        print('[BcLatent-Improve] Done with create_model')
