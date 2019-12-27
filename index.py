import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras_vggface.vggface import VGGFace

model = VGGFace(model='resnet50')
