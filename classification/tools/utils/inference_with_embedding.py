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
benchmark_path = 'data/dataset/labelled-pool/DataODAEL/test'
args = vars(parser.parse_args())


name = args["ckpt_path"]
cfg = load_config(args["config_path"])
ckpt_path = 'log/{}/{}'.format(name, 'model')
log_path = 'log/{}/model.h5'.format(name)

cfg.data_cfg.dataset_path = benchmark_path

data_path = glob.glob('{}/*/*'.format(cfg.data_cfg.dataset_path))
data_path = np.array(['/'.join(i.split('/')[-2:]) for i in data_path])

test_loader = make_data_loader(cfg.data_cfg, data_path, is_train=False, fetch_into_mem = True)

print(test_loader[0].shape)
model = build_model(cfg.data_cfg)
model.load_weights(log_path)

result = model.predict(test_loader[0])
import pickle
pickle.dump((result, test_loader[1], data_path), open('tools/embedding_result.pkl', 'wb'))