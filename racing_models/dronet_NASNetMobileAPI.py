from tensorflow.keras import Model
from tensorflow.python.keras.applications import NASNetMobile


class Dronet(Model):
    def __init__(self, num_outputs, include_top=True):
        super(Dronet, self).__init__()
        self.include_top = include_top
        self.create_model(num_outputs)

    def call(self, img):
        # Input
        model_d = self.nasNet(img)

        return model_d

    def create_model(self, num_outputs):
        print('[Dronet] Starting dronet with NASNetMobile')

        self.nasNet = NASNetMobile(include_top=False, weights="imagenet", classes=num_outputs)

        print('[Dronet] Done with dronet with NASNetMobile')