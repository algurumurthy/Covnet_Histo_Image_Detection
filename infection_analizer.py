from utils.downloaderapp import DownloaderApp
from pathology_detector import PathologyDetector
import os
from PIL import Image
import shutil
import json
from diagnosis import making_diagnosis
import time
from tqdm import tqdm
from visualizer import BloodCellVisualizer
from bloodcelldetector import BloodCellDetector
from convnet import ConvNet
import cv2
import argparse


class InfectionAnalizer:
    def __init__(self, slide_name, save_path, top=10):
        self.slide_name = slide_name
        self.save_path = save_path
        self.create_directory(save_path, slide_name + '_info')
        self.work_dir = os.path.join(save_path, slide_name + '_info')
        self.create_directory(os.path.join(save_path, slide_name + '_info'), 'downloads')
        self.create_directory(os.path.join(save_path, slide_name + '_info'), 'images')
        self.crawler = DownloaderApp(15, slide_name, self.work_dir)
        self.crawler.work_directory = os.path.join(self.work_dir, 'downloads')
        self.pathology_images = None
        self.top = top

    @staticmethod
    def remove_directory(directory_path, directory_name):
        if os.path.exists(os.path.join(directory_path, directory_name)):
            shutil.rmtree(os.path.join(directory_path, directory_name))

    @staticmethod
    def create_directory(directory_path, directory_name):
        if not os.path.exists(os.path.join(directory_path, directory_name)):
            os.makedirs(os.path.join(directory_path, directory_name))

    @staticmethod
    def conc_image(img_path, img_name, save_path, img_appx_size=1024):
        result = Image.new("RGB", (img_appx_size, img_appx_size))
        x, y = 0, 0
        for pic_name in sorted(os.listdir(os.path.join(img_path, img_name))):
            img = Image.open(os.path.join(img_path, img_name, pic_name))
            result.paste(img, (x, y, x + 256, y + 256))
            y += 256
            if y == img_appx_size:
                y = 0
                x += 256
        result.save(os.path.join(save_path, img_name))

    def download_slide(self):
        start_time = time.time()
        print('\033[94m* downloading the slide image in parts with low resolution (256x256) is in progress. Please wait...\033[0m')
        self.save_path = self.work_dir
        self.crawler.download_images()
        print('\033[92m* downloading was successfully completed in {:.2f} (s)\033[0m'.format(time.time()-start_time))

    def get_pathology_images(self, convnet_settings):
        start_time = time.time()
        print('\033[94m* searching for potentially infected parts. Please wait...\033[0m')
        detector = PathologyDetector(os.path.join(self.work_dir, self.slide_name), top=self.top)
        self.pathology_images = detector.find_pathology(convnet_settings)
        print('\033[92m* searching was successfully completed in {:.2f} (s)\033[0m'.format(time.time() - start_time))

    def download_images(self):
        start_time = time.time()
        print('\033[94m* downloading potentially infected parts in high resolution (1024x1024) is in progress. Please wait...\033[0m')
        for img_name in tqdm(self.pathology_images, ncols=80):
            self.crawler.approx_img(img_name)
        for img_name in os.listdir(self.crawler.work_directory):
            if '.jpeg' in img_name:
                self.conc_image(self.crawler.work_directory, img_name, os.path.join(self.work_dir, 'images'))
        print('\033[92m* downloading process was successfully completed in {:.2f} (s)\033[0m'.format(time.time() - start_time))

    def make_diagnosis(self, convnet_settings):
        start_time = time.time()
        print('\033[94m* making diagnosis is in progress. Please wait...\033[0m')
        image_diagnoses = []
        for img_name in os.listdir(os.path.join(self.work_dir, 'images')):
            if not img_name.startswith('.'):
                diagnosis = making_diagnosis(convnet_settings, os.path.join(self.work_dir, 'images', img_name))
                image_diagnoses.append((img_name, diagnosis))

        detector = BloodCellDetector('', '')
        convnet = ConvNet(convnet_settings)

        convnet.load_model(convnet_settings['model_path'])
        visualizer = BloodCellVisualizer(detector, convnet)

        image_diagnoses.sort(key=lambda x: x[1][0])
        image_diagnoses = image_diagnoses[::-1]

        cnt = 0
        for img_info in image_diagnoses[:self.top]:
            if img_info[1][1] == 'infection':
                img = cv2.imread(os.path.join(self.work_dir, 'images', img_info[0]))
                result_img = visualizer.highlight_cells(img,
                                                        labels=['1-lymphocytes', '2-neutrophils'],
                                                        show_scores=False,
                                                        make_zoom=False)
                cv2.imwrite(os.path.join(self.work_dir, 'marked' + img_info[0]), result_img)
                cnt += 1

        print('\033[92m* {} with selected neutrophil images are created in the "{}"\033[0m'.format(cnt, self.work_dir))
        print('\033[92m* making diagnosis was completed in {:.2f} (s)\033[0m'.format(time.time() - start_time))

        diagnoses = [d[1][1] for d in image_diagnoses]
        if 'infection' in diagnoses:
            total_diagnosis = 'infection'
        elif 'inflammation' in diagnoses:
            total_diagnosis = 'inflammation'
        else:
            total_diagnosis = 'bening'

        return image_diagnoses, total_diagnosis


def main(slide_name, save_path, pathology_settings, cell_classifier_settings):
    print('=' * 80)
    start_time = time.time()
    analizer = InfectionAnalizer(slide_name, save_path)
    analizer.download_slide()
    analizer.get_pathology_images(pathology_settings)
    analizer.download_images()
    diagnoses = analizer.make_diagnosis(cell_classifier_settings)
    print('=' * 80)
    print('* total elapsed time: {:.0f} (s)'.format(time.time() - start_time))
    print('-' * 80)

    for image_info in diagnoses[0]:
        if image_info[1][0] > 5:
            print(image_info[0] +
                  ' image information: ' + str(image_info[1][0]) +
                  ' neutrophils were founded,' +
                  ' diagnosis: ' + str(image_info[1][1]))
    print('=' * 80)

    diagnosis = diagnoses[1]
    if diagnosis == 'bening':
        color = '\033[92m'
    else:
        color = '\033[91m'

    print("Diagnosis: '\033[1m{color}{diagnosis}\033[0m'".format(diagnosis=diagnosis, color=color))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inflammation/infection detector')
    parser.add_argument('--slide_name', '-N', type=str, help='name of downloading slide')
    parser.add_argument('--save_path', '-P', type=str, help='save path')
    parser.add_argument('--cells_settings', '-C', type=str, help='path to a cells classifier convnet settings json')
    parser.add_argument('--pathology_settings', '-S', type=str, help='path to a cells pathology convnet settings json')
    args = parser.parse_args()
    main(args.slide_name, args.save_path, json.load(open(args.pathology_settings)),
         json.load(open(args.cells_settings)))
