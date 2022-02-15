import sys
sys.path.append('./')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import glob
from model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

from utils.utils import load_config
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
name = 'mitotic/baseline9'
cfg = load_config('config/mitotic.py')
ckpt_path = 'log/{}/{}'.format(name, 'model')
log_path = 'log/{}/model.h5'.format(name)

model = build_model(cfg.data_cfg)
model.load_weights(log_path)

import pickle
base_WSI = '../detection/mmdetection/mmdetection/data/mitotic/WSI/'
pkl_path = '../mitotic/data_generation/tools/result/dummy.pkl'
meta = pickle.load(open(pkl_path, 'rb'))

from tf_explain.core import GradCAM
import matplotlib.pyplot as plt

explainer = GradCAM()
import openslide
import cv2
from tqdm import tqdm
count = 0
for slide_name in meta:
    if(meta[slide_name]['set'] == 'test'):
        y_pred = meta[slide_name]['pred_bbox']
        slide = openslide.OpenSlide(base_WSI + slide_name )
        pred_new = []
        count+=1
#         if(count == 3): break
        for k in tqdm(y_pred):
#             print(k)
            xmin, ymin, xmax, ymax, actual_conf, det_conf = k
            cls_conf =  2 * (actual_conf - 0.5 * det_conf)
            x_center = int((xmin + xmax) / 2)
            y_center = int((ymin + ymax) / 2)

            if( cls_conf - det_conf > 0.4):
                img = np.array(slide.read_region((x_center - 64, y_center - 64), 0, (128, 128)))
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img = cv2.resize(img, (224, 224))
                img = tf.keras.preprocessing.image.img_to_array(img)
                data = ([img], None)
                
                grid = explainer.explain(data, model, class_index=0, image_weight = 0.0 )[..., 1]
                loc = np.unravel_index(np.argmax(grid, axis=None), grid.shape)
                dist = np.sqrt((112 - loc[0]) ** 2 + (112 - loc[1]) ** 2)
#                 print(np.argmax(grid[..., 0], axis = (0, 1)))
                if(dist > 70):
                    pred_new.append([xmin, ymin, xmax, ymax, cls_conf])
#                     print(det_conf, cls_conf, dist)
#                     plt.imshow(grid)
#                     plt.show()
#                     plt.imshow(np.array(img, dtype = np.uint8))
#                     plt.show()
                else:
                    pred_new.append([xmin, ymin, xmax, ymax, actual_conf])
            else:
                pred_new.append([xmin, ymin, xmax, ymax, actual_conf])
            meta[slide_name]['pred_bbox'] = pred_new
            
dst_pkl_path = '../mitotic/data_generation/tools/result/dummy2.pkl'
pickle.dump(meta, open(dst_pkl_path, 'wb'))