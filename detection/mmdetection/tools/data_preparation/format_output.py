import openslide
import os
import numpy as np
import pickle
import numpy as np
import sys
import pickle 
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Dataset Selection")
parser.add_argument("-m", "--modelpath", required=False, default=None, help="model path")
parser.add_argument("-c", "--configpath", required=False, default=None, help="relocation config path")
parser.add_argument("-i", "--inputpath", required=True, help="raw inference result path")
parser.add_argument("-o", "--outputpath", required=True, help="prediction output path")
parser.add_argument("-r", "--relocate", required=False, action="store_true", help="perform window relocation stage? default is False")
parser.add_argument("-s", "--set", required=False, default='test', help="split (train/test)")
args = vars(parser.parse_args())
dataset = args["dataset"]

output_path = args["outputpath"]
model_path = args["modelpath"]
input_path = args["inputpath"] 
config_path = args["configpath"]

relocate_window = args["relocate"]
split = args["set"]

print(relocate_window)
if(relocate_window):
    assert config_path is not None
    assert model_path is not None

if(dataset == 'CMC'):
    metadata = pickle.load(open('data/database/CMC_label.pkl', 'rb'))
elif(dataset == 'CCMCT'):
    metadata = pickle.load(open('data/database/CCMCT_label.pkl', 'rb'))
elif(dataset == 'menin'):
    metadata = pickle.load(open('data/database/menin_label.pkl', 'rb'))

else:
    raise Exception("Only CMC and CCMCT datasets are supported.")


if(split == 'test'):
    file_key = open('data/dataset/{}/test.txt'.format(dataset), "r")
elif(split == 'train'):
    file_key = open('data/dataset/{}/inference_train.txt'.format(dataset), "r")
pkl = pickle.load( open( input_path, "rb" ) )

if(relocate_window):
    f2 = open('data/dataset/{}/refocus.txt'.format(dataset), "wb")

for pred_bbox, file in tqdm(zip(pkl, file_key.readlines())):
    file_name, x, y = file.split('_')
    key =  file_name + '.svs'
    # print(key, metadata.keys())
    if(key not in metadata):
        continue
    if('pred_bbox' not in metadata[key]):
        metadata[key]['pred_bbox'] = []
    if(len(pred_bbox[0]) == 0):continue

    for bbox in pred_bbox[0]:
        area = (bbox[3] - bbox[1]) *  (bbox[2] - bbox[0])
        if(area == 0):continue

        xmin, ymin = bbox[0] + int(x), bbox[1] + int(y)
        xmax, ymax = bbox[2] + int(x), bbox[3] + int(y)
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        box_center_x = (bbox[2] + bbox[0]) / 2
        box_center_y = (bbox[3] + bbox[1]) / 2
        border = 25
        if(relocate_window):
            
            if( (box_center_x <= border or box_center_x >= 512 - border or box_center_y <= border or box_center_y >= 512 - border)):
                check_file = '{}_{}_{}\n'.format(file_name, int(x_center - 256), int(y_center - 256))
                f2.write(check_file.encode())
            else:
                conf = bbox[-1]
                metadata[key]['pred_bbox'].append([ xmin, ymin, xmax, ymax, conf])
        else:
            conf = bbox[-1]
            metadata[key]['pred_bbox'].append([ xmin, ymin, xmax, ymax, conf])
file_key.close()

if(relocate_window):
    f2.close()
    relocated_path = 'data/dataset/{}/refocus.pkl'.format(dataset)
    cmd = "python tools/test.py {} {} --out {}".format(config_path, model_path, relocated_path)
    os.system(cmd)

    f2 = open('data/dataset/{}/refocus.txt'.format(dataset), "r") 
    pkl = pickle.load( open( relocated_path, "rb" ) )
    
    for pred_bbox, file in tqdm(zip(pkl, f2.readlines())):
        file_name, x, y = file.split('_')
        key =  file_name + '.svs'
            
        if(len(pred_bbox[0]) == 0):continue
        for bbox in pred_bbox[0]:
            xmin, ymin = bbox[0] + int(x), bbox[1] + int(y)
            xmax, ymax = bbox[2] + int(x), bbox[3] + int(y)
            conf = bbox[-1]
            metadata[key]['pred_bbox'].append([ xmin, ymin, xmax, ymax, conf])

pickle.dump(metadata, open(output_path, 'wb'))
