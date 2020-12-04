from convnet import ConvNet
import tqdm
import os
import json
import cv2


path = '/Users/ayder/enviroments/pnenv/src/convnet/data/new_data/test'
err_path = '/Users/ayder/enviroments/pnenv/src/convnet/data/new_data/errors'
settings = json.load(open('/Users/ayder/enviroments/pnenv/src/convnet/settings/settings.json'))

print(settings)

settings['model_path'] = '/Users/ayder/enviroments/pnenv/src/convnet/models/model_is50_ep50_bs20'
convnet = ConvNet(settings)
convnet.load_model(settings['model_path'])

wrong_samples = {}
sample_cnt = 0
err_cnt = 0
for lbl in settings['labels']:
    lbl_path = os.path.join(path, lbl)
    files = [f for f in os.listdir(lbl_path) if not f.startswith('.')]
    wrong_samples[lbl] = []
    for f in tqdm.tqdm(files):
        sample_cnt += 1
        img = cv2.imread(os.path.join(lbl_path, f))
        prediction = convnet.predict(img)
        prediction_label = convnet.labels[prediction]
        if prediction_label != lbl:
            wrong_samples[lbl].append((f, prediction_label))
            err_cnt += 1
            err_file_path = os.path.join(err_path, lbl, '{}_like_{}{}'.format(os.path.splitext(f)[0], prediction_label, os.path.splitext(f)[1]))
            cv2.imwrite(err_file_path, img)

print(sample_cnt, err_cnt)
print(wrong_samples)
