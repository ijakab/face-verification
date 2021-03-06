import os
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from numpy.linalg import norm
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_PATH = '/Users/gauss/Projects/faks/osrv/vggface/'
NUMBER_OF_IMAGES = 4


def face_only(full):
    mtnc = MTCNN()
    results = mtnc.detect_faces(full)
    x1, y1, width, height = results[0]['box']
    return full[y1:y1+height, x1:x1+width]


def load_image(filename):
    loaded = pyplot.imread(BASE_PATH+filename)
    face = face_only(loaded)
    image = Image.fromarray(face)
    image = image.resize((250, 250))
    return asarray(image)


def get_feature_vectors(type):
    images = []
    for i in range(0, NUMBER_OF_IMAGES):
        loaded = load_image(type + '/' + str(i) + '.jpg')
        images.append(loaded)
    images = asarray(images, 'float32')
    processed = preprocess_input(images, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(250, 250, 3), pooling='avg')
    return model.predict(processed)


document_features = get_feature_vectors('documents')
current_features = get_feature_vectors('current')

for i in range(0, NUMBER_OF_IMAGES):
    matchRate = cosine(document_features[i], current_features[i])
    euclidianDistance = norm(document_features[i] - current_features[i])

    print('Slika ', i)
    print('Euklidska udaljenost je ', euclidianDistance)
    print('Kosinusova udaljenost: ', matchRate)
    if matchRate <= 0.5:
        print("Slike se podudaraju")
    else:
        print('Slike se ne podudaraju')
