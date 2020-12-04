from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
import numpy as np
import cv2
from scipy import misc
import argparse
import json


class ConvNet:
    def __init__(self, settings, model=None):
        self.labels = settings['labels']
        self.n_labels = len(self.labels)
        self.img_size = settings['img_size']
        self.model = self._init_model() if model is None else model

    def _init_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x * 1. / 255.,
                         input_shape=(self.img_size, self.img_size, 3),
                         output_shape=(self.img_size, self.img_size, 3)))
        model.add(Conv2D(32, (3, 3), input_shape=(self.img_size, self.img_size, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.7))
        model.add(Dense(self.n_labels))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    def load_model(self, model_file):
        self.model.load_weights(model_file)

    def predict(self, img, with_scores=False):
        img = misc.imresize(arr=img, size=(self.img_size, self.img_size, 3))
        img_arr = np.asarray([img])
        prediction = self.model.predict(img_arr)[0]
        scores = [round(score, 2) for score in prediction]
        label_index = np.where(scores == max(scores))[0][0]
        if with_scores:
            return label_index, scores
        else:
            return label_index


def main(settings, img_file):
    convnet = ConvNet(settings)
    convnet.load_model(settings['model_path'])
    img = cv2.imread(img_file)
    label_index, scores = convnet.predict(img, with_scores=True)
    print('Prediction: \033[1m\033[95m{cls}\033[0m (scores = {score})'.format(cls=convnet.labels[label_index].split('-')[1], score=scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train utility')
    parser.add_argument('--settings', '-S', type=str, help='path to the settings json')
    parser.add_argument('--img', '-I', type=str, help='path to the classifying image')
    args = parser.parse_args()
    main(json.load(open(args.settings)), args.img)
