import sys
sys.path.append('./')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
tf.get_logger().setLevel('ERROR')
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])

# ---------------------------path configuration------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--ckpt_path", required=True, help="Checkpoint directory")
parser.add_argument("-i", "--config_path", required=True, help="Path to config file")
parser.add_argument("-a", "--alpha", required=False, default=1, help="Alpha of focal loss")
parser.add_argument("-b", "--gamma", required=False, default=1, help="Gamma of focal loss")

args = vars(parser.parse_args())

name = args["ckpt_path"]
cfg = load_config(args["config_path"])
alpha = float(args["alpha"])
beta = float(args["gamma"])

target_dir = 'log/' + name + '/'
if not os.path.exists(target_dir): 
    os.makedirs(target_dir)

ckpt_path = 'log/{}/{}'.format(name, 'model')
log_path = 'log/{}'.format(name)

# ---------------------------fetch data------------------------

import pandas as pd
data = pd.read_csv('tools/mitotic_data.csv')
train_csv = data[data['set'] == 'train'].values[:, :4]
train_data, val_data = train_test_split(train_csv, train_size=0.95, random_state=42)
np.save('tools/val.npy', val_data)


train_loader = make_mitotic_loader(cfg.data_cfg, train_data, is_train=True)
val_loader = make_mitotic_loader(cfg.data_cfg, val_data, is_train=False)

model = build_model(cfg.data_cfg)
# # model.load_weights('log/mitotic/base_flip_br_rotate90_gblur_smooth-2-1e-4-cir-based/model.h5', by_name = True, skip_mismatch = True)

train_summary_writer = tf.summary.create_file_writer(log_path)

import tensorflow_addons as tfa
import tensorflow as tf

# l = tfa.losses.SigmoidFocalCrossEntropy(alpha = alpha, gamma = beta)
# l = symmetric_cross_entropy(alpha = alpha, beta = beta)

with train_summary_writer.as_default():
    do_train(cfg.data_cfg, model, train_loader, val_loader,
            scheduler=cfg.scheduler_cfg, ckpt_path=ckpt_path, loss_fn = None)

# test_loader = make_data_loader(cfg.data_cfg, val_data, is_train=False, fetch_into_mem = True)

# model.load_weights('{}.h5'.format(ckpt_path))
# evaluate(model, test_loader)
