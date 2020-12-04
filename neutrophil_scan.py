import argparse
import cv2
import json
from tqdm import tqdm
import os
import pandas as pd

from convnet import ConvNet
from bloodcelldetector import BloodCellDetector


def get_neutrophil_count(img, detector, convnet, score_thresh=0.75):
    detector.img = img
    detector.find_cells()
    neutr_count = 0
    for cell_img in tqdm(detector.cells_images, ncols=80):
        prediction, scores = convnet.predict(cell_img, with_scores=True)
        if max(scores) >= score_thresh:
            if prediction == 2:
                neutr_count += 1
    return neutr_count


def slide_diagnosis(path, convnet_settings):
    detector = BloodCellDetector('', '')
    convnet = ConvNet(convnet_settings)
    convnet.load_model(convnet_settings['model_path'])
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    stats = pd.DataFrame()
    columns = ['file', 'neutrophils']
    for f in tqdm(files, ncols=80):
        count = get_neutrophil_count(cv2.imread(os.path.join(path, f)), detector, convnet)
        stats = stats.append(pd.DataFrame([[f, count]], columns=columns))
    stats = stats.sort_values(by=['neutrophils'], ascending=[0])
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inflammation/infection detector')
    parser.add_argument('--settings', '-S', type=str, help='path to a settings json')
    parser.add_argument('--path', '-P', type=str, help='path to the image parts directory')
    parser.add_argument('--score-thresh', default=0.95, type=float, help='min prediction score threshold')
    args = parser.parse_args()
    files_vs_neutrophils = slide_diagnosis(args.path, json.load(open(args.settings)))
    print('\n', files_vs_neutrophils.to_string(index=False))
