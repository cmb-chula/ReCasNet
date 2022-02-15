import pickle
import numpy as np
import openslide, cv2
import pickle
import numpy as np
import sys
from sklearn.neighbors import KDTree
import pandas as pd
from tqdm import tqdm

def show_im(x_center, y_center, slide):
    import matplotlib.pyplot as plt
    img = np.array(slide.read_region((x_center - 64, y_center - 64), 0, (128, 128)))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img[..., ::-1]


def process_nms(data, radius = 25):
    scores = data[:, -1]
    y_pred = data[:, :-1]
    y_pred, scores = non_max_suppression_by_distance(y_pred, scores, radius = radius)
    return np.concatenate([y_pred, scores[..., None]], axis = 1)
def non_max_suppression_by_distance(boxes, scores, radius: float = 25, det_thres=None):
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

dataset = 'CCMCT'
# metadata = pickle.load( open( '../detection/mmdetection/test_pred/train1ccmct.pkl', "rb" ) )
# metadata = pickle.load( open( './cls1CCMCT_train.pkl', "rb" ) )
metadata = pickle.load( open('./cls1CCMCT_trainv2.pkl', "rb") )

method = 'disagreement'
base_path = '../detection/mmdetection/data/dataset/{}/WSI/'.format(dataset)
num_query = 20000

df =  pd.read_csv('../detection/mmdetection/data/database/{}_trainlabel.csv'.format(dataset))
df = df[(df['set'] == 'train')].values
meta_b = {}
for i in df:
    name, x_center, y_center, _, _ = i
    name += '.svs'
    if(name not in meta_b):
        meta_b[name] = []
    meta_b[name].append(np.array([x_center - 25, y_center - 25, x_center + 25, y_center + 25]))

def query(metadata, perform_query = False, thresh = 0, method = 'disagreement', num_query = 20000):
    stat = []
    for i in metadata:
        if(metadata[i]['set'] != 'train'): continue
        slide = openslide.OpenSlide(base_path + i )
        if('pred_bbox' not in metadata[i]): continue
        y_pred = np.array(metadata[i]['pred_bbox'], dtype = np.object)
        y_pred =  process_nms(y_pred, radius = 25)

        y_true2 = np.array(meta_b[i])
        print(i, len(y_true2))
        true2_center_x = (y_true2[:,0] + y_true2[:, 2]) / 2
        true2_center_y = (y_true2[:,1] + y_true2[:, 3]) / 2
        true2_center = np.concatenate([true2_center_x[...,None], true2_center_y[..., None]], axis = 1)
        for a in  tqdm(y_pred):
            xmin, ymin, xmax, ymax, original, cls_conf = a
#             if(det_conf < 0.047) : continue
            pred_center_x = (xmin + xmax) / 2
            pred_center_y = (ymin + ymax) / 2
            det_conf = original[-1]
            
            pred_center = np.array([pred_center_x, pred_center_y])[None, ]
            dist = pred_center - true2_center
            dist = np.sqrt(dist[:,0] ** 2 + dist[:, 1] ** 2)
            loc = np.argmin(dist)
            min_dist2 =  np.min(dist)
            if(min_dist2 <= 25): continue # ignore positive cls
            criterion = None

            if(method == 'disagreement'):
                criterion = lambda x : np.abs(x[0] -x[1]) 
            elif(method == 'entropy'):
                criterion = lambda x : -np.log(x[0])

            if(not perform_query):
                stat.append(criterion((cls_conf, det_conf)))
            else:
                if(criterion((cls_conf, det_conf)) > thresh):
                    file_name = 'D/{}_{}_{}.png'.format(i[:-4], str(int(pred_center_x)), str(int(pred_center_y)))
                    img = show_im(int(pred_center_x), int(pred_center_y), slide)
                    cv2.imwrite(file_name, img)

    return stat

stat = query(metadata, method = method, num_query = num_query)
print("--->", len(stat), sorted(stat, reverse = True)[0], sorted(stat, reverse = True)[-1])
thresh = sorted(stat, reverse = True)[num_query]
query(metadata, perform_query = True, thresh = thresh, method = method, num_query = num_query)