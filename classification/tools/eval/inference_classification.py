import sys
sys.path.append('./')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import cv2
import pickle
import openslide
from tqdm import tqdm
from data.cls_inference_server import ClassificationInferenceServer
import time
from multiprocessing.dummy import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Dataset Selection")
parser.add_argument("-ip", "--input_pkl_path", required=True, help="input metadata path")
parser.add_argument("-op", "--output_pkl_path", required=True, help="output metadata path")
parser.add_argument("-m", "--model_path", required=True, help="classification model path")
parser.add_argument("-c", "--cls_weight", required=False, default=1, help="confidence weight of classification model (1 - omega)")
parser.add_argument("-s", "--split", required=False, default='test', help="split (test/train)")

args = vars(parser.parse_args())
dataset = args["dataset"]
input_pkl_path = args["input_pkl_path"]
output_pkl_path = args["output_pkl_path"]
model_path = args["model_path"]
cls_weight = float(args["cls_weight"])
assert 0 <= cls_weight <= 1 

WSI_path = '../detection/mmdetection/data/dataset/{}/'.format(dataset)

mode = args["split"]
seek_data = False if mode == 'test' else True
metadata = pickle.load( open( input_pkl_path, "rb" ) )
model = tf.saved_model.load('converted_model/{}'.format(model_path))

def push_to_server(data):
    xmin, ymin, xmax, ymax, conf = data
    x_center = int((xmin + xmax) / 2)
    y_center = int((ymin + ymax) / 2)
    img = np.array(slide.read_region((x_center - 64, y_center - 64), 0, (128, 128)), dtype = np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)#[:,:,0:3]
    cls_server.push([xmin, ymin, xmax, ymax, conf, img])

num_slide = len(metadata)
for idx, slide_name in enumerate(metadata):
    print("slide {}/{} : {}".format(idx + 1, num_slide, slide_name))
    if(metadata[slide_name]['set'] == mode):
        cls_server = ClassificationInferenceServer(model, cls_weight = cls_weight, seek_data = seek_data)
        cls_server.setDaemon(True)
        cls_server.start()

        y_pred = np.array(metadata[slide_name]['pred_bbox'])
        y_pred_new = []
        slide = openslide.OpenSlide(WSI_path + 'WSI/'+ slide_name)

        start_time = time.time()
        p = Pool(8)
        p.map(push_to_server, y_pred)
        p.close()
        cls_server.push(None)
        metadata[slide_name]['pred_bbox'] = cls_server.get_prediction_result()
        print("Inference time = {}".format(time.time() - start_time))

pickle.dump(metadata, open(output_pkl_path, 'wb'))
