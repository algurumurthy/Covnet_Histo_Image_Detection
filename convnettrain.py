import numpy as np
import cv2
from scipy import misc
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import keras
import os
import pandas as pd
from tqdm import tqdm
from convnet import ConvNet
import argparse
import json


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs=None):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))


class ConvNetTrain:
    def __init__(self, convnet, settings):
        self.convnet = convnet
        self.data_path = settings['data_path']
        self.epochs = settings['epochs']
        self.batch_size = settings['batch_size']
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.history = AccuracyHistory()

    def load_dataset(self, path):
        x, y = [], []
        for label in self.convnet.labels:
            imgs_dir = os.path.join(path, label)
            if os.path.exists(imgs_dir):
                for img_file in tqdm(os.listdir(imgs_dir), ncols=80,
                                     desc="'{}/{}'".format(os.path.basename(imgs_dir), os.path.basename(path))):
                    img = cv2.imread(os.path.join(path, label, img_file))
                    if img is not None:
                        img = misc.imresize(arr=img, size=(self.convnet.img_size, self.convnet.img_size, 3))
                        img_arr = np.asarray(img)
                        x.append(img_arr)
                        y.append(label)
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def prepare_data(self):
        train_path = os.path.join(self.data_path, 'train')
        test_path = os.path.join(self.data_path, 'test')
        self.X_train, self.y_train = self.load_dataset(train_path)
        self.X_test, self.y_test = self.load_dataset(test_path)
        encoder = LabelEncoder()
        encoder.fit(self.y_train)
        self.y_train = encoder.transform(self.y_train)
        self.y_test = encoder.transform(self.y_test)
        self.y_train = to_categorical(self.y_train, len(self.convnet.labels))
        self.y_test = to_categorical(self.y_test, len(self.convnet.labels))
        print('Training samples: {}\nTest samples: {}'.format(len(self.X_train), len(self.X_test)))

    def train(self):
        self.prepare_data()
        self.convnet.model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.2,
            epochs=self.epochs,
            shuffle=True,
            batch_size=self.batch_size,
            callbacks=[self.history])

    def load_model(self, model_file):
        self.convnet.model.load_weights(model_file)

    def save_model(self, model_file):
        self.convnet.model.save_weights(model_file)

    def summary(self):
        prediction = self.convnet.model.predict(self.X_test)
        prediction = np.array([np.where(predict == max(predict))[0][0] for predict in prediction])
        y_test = np.array([np.where(predict == max(predict))[0][0] for predict in self.y_test])
        stats = pd.DataFrame()
        columns = ["p", "n", "tp", "tn", "fp", "fn", "tpr", "tnr", "sensitivity", "specificity", "acc", "f1_score"]
        for i in range(len(self.convnet.labels)):
            p = len(np.where(y_test == i)[0])
            n = len(np.where(y_test != i)[0])
            tp = len(np.intersect1d(np.where(prediction == i), np.where(y_test == i)))
            tn = len(np.intersect1d(np.where(prediction != i), np.where(y_test != i)))
            fp = len(np.intersect1d(np.where(prediction == i), np.where(y_test != i)))
            fn = len(np.intersect1d(np.where(prediction != i), np.where(y_test == i)))
            tpr = tp / p
            tnr = tn / n
            sensitivity = tp / (tp + fn)
            specificity = tn / (fp + tn)
            acc = (tp + tn) / (tp + fp + tn + fn)
            f1_score = 2*tp / (2*tp + fp + fn)
            stats = stats.append(pd.DataFrame([[p, n, tp, tn, fp, fn,
                                                round(tpr, 4),
                                                round(tnr, 4),
                                                round(sensitivity, 4),
                                                round(specificity, 4),
                                                round(acc, 4),
                                                round(f1_score, 4)]],
                                              columns=columns))
        stats.index = self.convnet.labels

        print(self.history.acc)
        print(self.history.val_acc)

        return stats


def main(settings):
    convnet = ConvNet(settings)
    train = ConvNetTrain(convnet, settings)
    train.train()
    model_path = settings['model_path']
    train.save_model(model_path)
    summary = train.summary()
    print(summary)
    summary.to_csv("{}_summary.csv".format(os.path.splitext(model_path)[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train utility')
    parser.add_argument('--settings', '-S', type=str, help="path to a settings json")
    args = parser.parse_args()
    main(json.load(open(args.settings)))
