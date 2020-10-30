from tensorflow.keras import Model
from tensorflow.python.keras.applications import MobileNetV2


class Dronet(Model):
    def __init__(self, num_outputs, include_top=True):
        super(Dronet, self).__init__()
        self.include_top = include_top
        self.create_model(num_outputs)

    def call(self, img):
        # Input
        model_d = self.mobile(img)
        return model_d

    def create_model(self, num_outputs):
        print('[Dronet] Starting dronet with MobileNetV2')

        self.mobile = MobileNetV2(include_top=self.include_top, weights=None, classes=num_outputs)

        print('[Dronet] Done with dronet with MobileNetV2')