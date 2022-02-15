import sys
sys.path.append('./')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import glob
from engine.evaluate import evaluate
from engine.trainer import do_train
from data.build import make_mitotic_loader
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
parser.add_argument("-o", "--ckpt_path", required=True, help="Checkpoint directory")
parser.add_argument("-i", "--config_path", required=True, help="Path to config file")
benchmark_path = 'data/dataset/labelled-pool/DataODAEL/test'
args = vars(parser.parse_args())


name = args["ckpt_path"]
cfg = load_config(args["config_path"])
ckpt_path = 'log/{}/{}'.format(name, 'model')
log_path = 'log/{}/model.h5'.format(name)

# cfg.data_cfg.dataset_path = benchmark_path

import pandas as pd
data = pd.read_csv('tools/mitotic_data.csv')
test_csv = data[data['set'] == 'test'].values[:, :4]


test_loader = make_mitotic_loader(cfg.data_cfg, test_csv, is_train=False)
# print(test_loader[0].shape)
model = build_model(cfg.data_cfg)
model.load_weights(log_path)

# evaluate(model, test_loader)

from tqdm import tqdm
pbar = tqdm()

y_true, y_pred = [], []d
while(True):
    test_data = test_loader.grab()
    if(test_data is None): break
    X_test, Y_test = test_data
    y_true += list(Y_test)
    y_pred += list(model(X_test))
    pbar.update(1)
pbar.close()

y_true = np.argmax(np.array(y_true), axis = 1)
y_pred = np.argmax(np.array(y_pred), axis = 1)

y_pred[y_pred > 1] = 1
y_true[y_true > 1] = 1

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
print(classification_report(y_true, y_pred, digits=4))
print(confusion_matrix(y_true, y_pred))

print(y_true.shape, y_pred.shape)

