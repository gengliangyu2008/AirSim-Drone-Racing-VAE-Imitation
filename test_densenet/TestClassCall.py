
class Derivative(object):
    def __init__(self, f):
        self.f = f

    '''
    def __call__(self, x):
        return self.f * x
    '''

    def call(self, x):
        return self.f ** x


if __name__ == "__main__":

    d1 = Derivative(10)
    print(d1(4))