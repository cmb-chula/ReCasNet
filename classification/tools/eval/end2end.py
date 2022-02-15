import pickle
import numpy as np
import openslide, cv2
import pickle
import numpy as np
import sys
from sklearn.neighbors import KDTree

def non_max_suppression_by_distance(boxes, scores, radius: float = 25, det_thres=None):
    ''' 
    The NMS code is from https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/lib/nms_WSI.py.
    '''
    if (det_thres is not None): # perform thresholding
        to_keep = scores>det_thres
        boxes = boxes[to_keep]
        scores = scores[to_keep]
    
    if boxes.shape[-1]==4: # BBOXES
        center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
        center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
    else:
        center_x = boxes[:, 0]
        center_y = boxes[:, 1]
        
    X = np.dstack((center_x, center_y))[0]
    tree = KDTree(X)

    sorted_ids = np.argsort(scores)[::-1]

    ids_to_keep = []
    ind = tree.query_radius(X, r=radius)

    while len(sorted_ids) > 0:
        id = sorted_ids[0]
        ids_to_keep.append(id)
        sorted_ids = np.delete(sorted_ids, np.in1d(sorted_ids, ind[id]).nonzero()[0])

    return boxes[ids_to_keep], scores[ids_to_keep]

def process_nms(data, radius = 25):
    scores = data[:, 4]
    y_pred = data[:, :4]
    y_pred, scores = non_max_suppression_by_distance(y_pred, scores, radius = radius)
    return np.concatenate([y_pred, scores[..., None]], axis = 1)

def acquire_pred_key(WSI_path, metadata, det_thresh, mode = 'GB'):
    pred_key = {}
    loc_key = {}

    for i in metadata:
        if(metadata[i]['set'] not in  'test'): continue
        slide = openslide.OpenSlide(WSI_path + i)
        width, height = slide.dimensions 
        heatmap = np.zeros((height // 32, width // 32))
        
        y_pred = np.array(metadata[i]['pred_bbox'])
        y_pred =  process_nms(y_pred, radius = 25)
        y_true = np.array(metadata[i]['bbox'])
        true_center_x = (y_true[:,0] + y_true[:, 2]) / 2
        true_center_y = (y_true[:,1] + y_true[:, 3]) / 2
        true_center = np.concatenate([true_center_x[...,None], true_center_y[..., None]], axis = 1)
        match_idx = np.zeros(len(y_true))
        for box in y_pred:
            xmin, ymin, xmax, ymax, det_conf = box
            pred_center_x = (xmin + xmax) / 2
            pred_center_y = (ymin + ymax) / 2
            pred_center = np.array([pred_center_x, pred_center_y])[None, ]

            if(det_conf > det_thresh):
                xmin = int((pred_center[0, 0] - (WINDOW_W /2)) // 32)
                xmax = int((pred_center[0, 0] + (WINDOW_W /2)) // 32)
                ymin = int((pred_center[0, 1] - (WINDOW_H /2)) // 32)
                ymax = int((pred_center[0, 1] + (WINDOW_H /2)) // 32)                
                heatmap[max(ymin, 0) : min(ymax, height // 32), max(xmin, 0) : min(xmax, width // 32)] += 1

        
        argmax2D = np.unravel_index(heatmap.argmax(), heatmap.shape)
        if(mode == 'GA'):
            pred_key[i] = heatmap[argmax2D]
        elif(mode == 'GB'):
            pred_key[i] = GT_heatmap[i][argmax2D]
        loc_key[i] = argmax2D

    return pred_key, loc_key


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ip", "--pkl_path", required=True, help="pkl metadata path")
parser.add_argument("-d", "--dataset", required=True, help="Dataset Selection")
parser.add_argument("-s", "--setting", required=True, help="setting (GA or GB)")

args = vars(parser.parse_args())
eval_path = args["pkl_path"]
dataset = args["dataset"]
mode = args["setting"]

metadata = pickle.load( open( eval_path, "rb" ) )

WSI_path = '../detection/mmdetection/data/dataset/{}/WSI/'.format(dataset)

slides = []
ans_key = {}
GT_heatmap = {}
WINDOW_H = 5333
WINDOW_W = 7110 
# generate GT Mitotic count for each slide
for slide_name in metadata:
    if(metadata[slide_name]['set'] != 'test'): 
        continue

    slides.append(slide_name)
    slide = openslide.OpenSlide(WSI_path + slide_name)
    width, height = slide.dimensions 
    heatmap = np.zeros((height // 32, width // 32))
    y_true2 = np.array(metadata[slide_name]['bbox'])
    for k in range(len(y_true2)):
        y_true2_idx = y_true2[k]
        xmin = int((y_true2_idx[0] - (WINDOW_W /2)) // 32)
        xmax = int((y_true2_idx[0] + (WINDOW_W /2)) // 32)
        ymin = int((y_true2_idx[1] - (WINDOW_H /2)) // 32)
        ymax = int((y_true2_idx[1] + (WINDOW_H /2)) // 32)

        heatmap[max(ymin, 0) : min(ymax, height // 32), max(xmin, 0) : min(xmax, width // 32)] += 1
    ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    ans_key[slide_name] = int(heatmap.max())
    GT_heatmap[slide_name] = heatmap

true_ans = np.array([ans_key[i] for i in slides])
print(true_ans)
min_MAPE = np.inf
res = None
for thresh in np.arange(0.1, 0.9, 0.01):
    pred_key, loc_key = acquire_pred_key(WSI_path, metadata, thresh, mode = mode)
    pred_ans = np.array([pred_key[i] for i in slides])
    l1 = np.abs(true_ans - pred_ans)
    MAPE = (l1  / true_ans)
    if(MAPE.mean() < min_MAPE):
        min_MAPE = MAPE.mean()
        res = [thresh, l1.mean(), l1.std(), MAPE.mean(), MAPE.std()]
    print("thresh = {:.2f} MAE = {:.3f} SD = {:.3f}, MAPE = {:.3f}, MAPE SD = {:.3f}".format(thresh, l1.mean(),  l1.std(), 100 * MAPE.mean(), MAPE.std() ))
    print(true_ans)
    print(pred_ans)
print("BEST MAPE")
print("thresh = {:.2f} MAE = {:.3f} SD = {:.3f}, MAPE = {:.3f}, MAPE SD = {:.3f}".format(res[0], res[1], res[2], 100 * res[3], res[4]))
