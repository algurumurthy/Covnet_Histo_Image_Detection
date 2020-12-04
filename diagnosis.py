import argparse
import cv2
import json
from tqdm import tqdm
import os

from convnet import ConvNet
from bloodcelldetector import BloodCellDetector
from visualizer import BloodCellVisualizer


def making_diagnosis(convnet_settings, img_path, score_thresh=0.97, infection_threshold=5, inflammation_threshold=5, highlight=False):
    print(img_path)
    img = cv2.imread(img_path)

    # select blood cells
    detector = BloodCellDetector('', '')
    detector.img = img
    detector.find_cells()

    # print("Found {} blood cell".format(len(detector.cells_images)))

    # classify the founded blood cells
    convnet = ConvNet(convnet_settings)
    convnet.load_model(convnet_settings['model_path'])
    lymph_count = 0
    neutr_count = 0
    total_count = 0
    for cell_img in tqdm(detector.cells_images, ncols=100):
        total_count += 1
        prediction, scores = convnet.predict(cell_img, with_scores=True)
        if max(scores) >= score_thresh:
            if prediction == 1:
                lymph_count += 1
            elif prediction == 2:
                neutr_count += 1

    print("{} lymphocytes and {} neutrophils were found among {} blood cells".format(lymph_count, neutr_count, total_count))

    # making diagnosis
    if neutr_count > infection_threshold:
        diagnosis = 'infection'
    elif lymph_count > inflammation_threshold:
        diagnosis = 'inflammation'
    else:
        diagnosis = 'bening'

    if highlight:
        visualizer = BloodCellVisualizer(detector, convnet)
        result_img = visualizer.highlight_cells(img, labels=['1-lymphocytes', '2-neutrophils'], show_scores=False, make_zoom=True)
        output_path = '{}_diagnosis_{}{}'.format(os.path.splitext(img_path)[0], diagnosis, os.path.splitext(img_path)[1])
        cv2.imwrite(output_path, result_img)
        print('File "{}" was created.'.format(output_path))

    return neutr_count, diagnosis


def main(img, settings, score_thresh):
    _, diagnosis = making_diagnosis(json.load(open(settings)), img, score_thresh=score_thresh, highlight=True)
    if diagnosis == 'bening':
        color = '\033[92m'
    else:
        color = '\033[91m'

    print("Diagnosis: '\033[1m{color}{diagnosis}\033[0m'".format(diagnosis=diagnosis, color=color))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inflammation/infection detector')
    parser.add_argument('--settings', '-S', type=str, help='path to a settings json')
    parser.add_argument('--img', '-I', type=str, help='path to the image')
    parser.add_argument('--score-thresh', default=0.95, type=float, help='min prediction score threshold')
    args = parser.parse_args()
    main(args.img, args.settings, score_thresh=args.score_thresh)
