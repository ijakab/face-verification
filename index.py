import os
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from keras_vggface.vggface import VGGFace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_PATH = '/Users/gauss/Projects/faks/osrv/vggface'
IMAGE_SIZE = (250, 250)


def load_image(filename):
    loaded = pyplot.imread(BASE_PATH+filename)
    image = Image.fromarray(loaded)
    image = image.resize(IMAGE_SIZE)
    return asarray(image)


img = load_image('/current/demo.jpg')
print(img.shape)
print(img)
