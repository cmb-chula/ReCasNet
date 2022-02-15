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
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])

# ---------------------------path configuration------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--ckpt_path", required=True, help="Checkpoint directory")
parser.add_argument("-i", "--config_path", required=True, help="Path to config file")
parser.add_argument("-s", "--src_path", required=True, help="Path to data source")
parser.add_argument("-d", "--dst_path", required=True, help="Inference result path")


args = vars(parser.parse_args())
name = args["ckpt_path"]
cfg = load_config(args["config_path"])
src_path = args["src_path"]
dst_path = args["dst_path"]


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])

ckpt_path = 'log/{}/{}'.format(name, 'model')
log_path = 'log/{}/model.h5'.format(name)
cfg.data_cfg.dataset_path = src_path

# data_path = np.array(['/'.join(i.split('/')[-2:]) for i in data_path])

X, path = make_data_loader(cfg.data_cfg, os.listdir(src_path), is_train=False, fetch_into_mem = True, get_label = False)
print(X.shape)
model = build_model(cfg.data_cfg)
model.load_weights(log_path)

y_pred = np.argmax(model.predict(X, verbose = 1), axis = 1)

try:    os.makedirs(dst_path)
except: pass

for filename, y in zip(path, y_pred):
    import shutil 
    result_dir = f"{dst_path}/{str(y)}"
    try:    os.makedirs(result_dir)
    except: pass
    shutil.copyfile(f"{src_path}/{filename}", f"{result_dir}/{filename}")

#python3 tools/inference.py -i config/gender.py -o gender/17-Jun-2020 -s data/dataset/raw/CTW/ctw-001 -d data/dataset/raw/CTW/ctw-001-out