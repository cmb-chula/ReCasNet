import pickle
import openslide
from pascal_voc_writer import Writer
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import argparse, os


        
def get_intersection(query, db, overlap_dist):
    dist = query - db 
    dist = np.maximum(np.abs(dist[:,0]), np.abs(dist[:, 1])) # distance at inf norm (square)
    res = np.where(dist <= overlap_dist)[0]
    return res        

def annotate(file, slide_name, query_center, gt_bbox, gt_bbox_center, overlap_dist):
    ''' Generate XML label from w.r.t. the sample patch center of size 512 x 512
    '''
    top_x, top_y = query_center[0] - overlap_dist, query_center[1] - overlap_dist
    intersected_idx = get_intersection(query_center, gt_bbox_center, overlap_dist)   
    intersected_object = gt_bbox[intersected_idx]
    xml_path = 'Annotations/' + slide_name.split('.')[0] + '_{}_{}.xml'.format(top_x, top_y)
    writer = Writer(xml_path, int(overlap_dist*2), (overlap_dist*2))
    for bbox in  intersected_object:
        xmin ,ymin, xmax, ymax = bbox
        bbox_xmin, bbox_ymin = xmin - query_center[0] + overlap_dist, ymin - query_center[1] + overlap_dist 
        bbox_xmax, bbox_ymax = bbox_xmin + 50, bbox_ymin + 50
        writer.addObject('mitotic', bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)

    line_name = '{}_{}_{}\n'.format(slide_name.split('.')[0], top_x, top_y)
    writer.save(xml_path)
    f.write(line_name.encode())

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Dataset Selection")
parser.add_argument("-n", "--nbsample", required=False, default=5000, help="Number of sampled window per WSI")
parser.add_argument("-hn", "--hnratio", required=False, default=0.4, help="Proportion of window containing at least one hard-negative object")
parser.add_argument("-r", "--rratio", required=False, default=0.1, help="Proportion of random window")
parser.add_argument("-w", "--windowsize", required=False, default=512, help="Window size")


args = vars(parser.parse_args())
dataset = args["dataset"]
nb_sample = args["nbsample"]
hard_neg_ratio = args["hnratio"]
random_ratio = args["rratio"]
overlap_dist = int(args["windowsize"] / 2)

if(dataset == 'CMC'):
    metadata = pickle.load(open('data/database/CMC_label.pkl', 'rb'))
    WSI_path = 'data/dataset/CMC/WSI/'
    df = pd.read_csv('data/database/CMC_trainlabel.csv')
elif(dataset == 'CCMCT'):
    metadata = pickle.load(open('data/database/CCMCT_label.pkl', 'rb'))
    WSI_path = 'data/dataset/CCMCT/WSI/'
    df = pd.read_csv('data/database/CCMCT_trainlabel.csv')
else:
    raise Exception("Only CMC and CCMCT datasets are supported.")

try: os.makedirs('Annotations/') 
except: pass

for i in df.values:
    name, x_center, y_center, cls, sets = i
    name += '.svs'
    if(cls in ['Mitosislike', 'Nonmitosis']):
        xmin, xmax = x_center - 25, x_center + 25
        ymin, ymax = y_center - 25, y_center + 25
        if('hard_neg' not in metadata[name]):
            metadata[name]['hard_neg'] = []
        metadata[name]['hard_neg'].append(np.array([xmin, ymin, xmax, ymax]))
        
print("Generate train text file")
f = open('train.txt', "wb")
for slide_name in metadata:
    if(metadata[slide_name]['set'] == 'test'):continue
    slide = openslide.OpenSlide(WSI_path + slide_name)
    width, height = slide.dimensions 
    print(slide_name, width, height)
    gt_bbox = np.array(metadata[slide_name]['bbox'])
    gt_bbox_x = (gt_bbox[:, 2] + gt_bbox[:, 0])/2
    gt_bbox_y = (gt_bbox[:, 3] + gt_bbox[:, 1])/2
    gt_bbox_center = np.concatenate([gt_bbox_x[...,None], gt_bbox_y[...,None]], axis = 1)
    
    hard_bbox = np.array(metadata[slide_name]['hard_neg'], dtype = np.int)
    hard_bbox_x = (hard_bbox[:, 2] + hard_bbox[:, 0])/2
    hard_bbox_y = (hard_bbox[:, 3] + hard_bbox[:, 1])/2
    hard_bbox_center = np.concatenate([hard_bbox_x[...,None], hard_bbox_y[...,None]], axis = 1)
    
    for _ in range(int(nb_sample * random_ratio)):
        query_center = np.array([np.random.randint(256, width), np.random.randint(256, height)])
        annotate(f, slide_name, query_center, gt_bbox, gt_bbox_center, overlap_dist)
        
    for _ in range(int(nb_sample * hard_neg_ratio)):
        query_center = np.array(hard_bbox_center[np.random.randint(len(hard_bbox_center))], dtype = np.int) + np.random.randint(-192, 192, 2)
        annotate(f, slide_name, query_center, gt_bbox, gt_bbox_center, overlap_dist)

    for _ in range(int(nb_sample * (1 - (random_ratio + hard_neg_ratio)))):
        query_center = np.array(gt_bbox_center[np.random.randint(len(gt_bbox_center))], dtype = np.int) + np.random.randint(-192, 192, 2)
        annotate(f, slide_name, query_center, gt_bbox, gt_bbox_center, overlap_dist)
        
f.close()

print("Generate test text file")
f2 = open('test.txt', "wb")
for slide_name in metadata:
    if(metadata[slide_name]['set'] == 'train'):continue
    slide = openslide.OpenSlide(WSI_path + slide_name)
    width, height = slide.dimensions 
    for i in range(0, width, 512):
        for j in range(0, height, 512):
            f2.write('{}_{}_{}\n'.format(slide_name.split('.')[0], str(i), str(j)).encode())

f2.close()

print("Generate inference train text file")
f2 = open('inference_train.txt', "wb")
for slide_name in metadata:
    if(metadata[slide_name]['set'] == 'test'):continue
    slide = openslide.OpenSlide(WSI_path + slide_name)
    width, height = slide.dimensions 
    for i in range(0, width, 512):
        for j in range(0, height, 512):
            f2.write('{}_{}_{}\n'.format(slide_name.split('.')[0], str(i), str(j)).encode())

f2.close()
