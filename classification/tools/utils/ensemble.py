import sys
sys.path.append('./')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import glob
from engine.evaluate import evaluate
from engine.trainer import do_train
from data.build import make_data_loader
from model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from utils.utils import load_config
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
# ---------------------------path configuration------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--config_path", required=True, help="Path to config file")


benchmark_path = 'data/dataset/labelled-pool/DataODAEL/test'
args = vars(parser.parse_args())

cfg = load_config(args["config_path"])
cfg.data_cfg.dataset_path = benchmark_path

data_path = glob.glob('{}/*/*'.format(cfg.data_cfg.dataset_path))
data_path = np.array(['/'.join(i.split('/')[-2:]) for i in data_path])

model = build_model(cfg.data_cfg)

y_pred = []
path = ['mitotic/l.a=.75b=0.2', 'mitotic/l.a=.25b=1', 'mitotic/l.a=.25b=2']
for model_dir in path:
    test_loader = make_data_loader(cfg.data_cfg, data_path, is_train=False, fetch_into_mem = True)
    model.load_weights("log/{}/model.h5".format(model_dir))
    y_true, pred = evaluate(model, test_loader)
    y_pred.append(pred)
    del test_loader
    import gc
    n = gc.collect()
    print("Number of unreachable objects collected by GC:", n)

y_pred = np.argmax(np.array(y_pred, dtype = np.float32).mean(axis = 0), axis = 1)
y_pred[y_pred > 1]= 1
y_true[y_true > 1]= 1

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred, digits = 4))
print(confusion_matrix(y_true, y_pred))