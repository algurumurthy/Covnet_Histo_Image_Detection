from convnet import ConvNet
import os
import cv2
import numpy as np
from scipy import misc
from tqdm import tqdm


class PathologyDetector:
    def __init__(self, imgs_directory, top=10):
        self.imgs_directory = imgs_directory
        self.top = top

    def find_pathology(self, convnet_settings):
        detection = ConvNet(convnet_settings)
        detection.load_model(convnet_settings['model_path'])
        pathology_images = []
        for img_name in tqdm(os.listdir(self.imgs_directory), ncols=80):
            if '.jpeg' not in img_name:
                continue
            img = cv2.imread(os.path.join(self.imgs_directory, img_name))
            img = misc.imresize(arr=img, size=(detection.img_size, detection.img_size, 3))
            img_arr = np.asarray([img])
            prediction = detection.model.predict(img_arr)[0][0]
            if int(prediction) == 0:
                pathology_images.append((img_name, prediction))

        pathology_images = np.asarray(pathology_images)
        pathology_images = pathology_images[pathology_images[:, 1].argsort()]

        pathology_images = pathology_images[::-1]
        pathology_images = [img_name for img_name, _ in pathology_images]

        return pathology_images[:self.top]
