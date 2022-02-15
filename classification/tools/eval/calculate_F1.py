import os
import numpy as np
np.random.seed(0)
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

def getf1(metadata, det_tresh, mode = 'test'):
    '''
    Evaluation code was based on https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/lib/calculate_F1.py.
    '''
    sum_GT, sum_TP, sum_FP, sum_FN = 0, 0, 0 , 0
    from sklearn.neighbors import KDTree
    for slide_name in metadata:
        if(metadata[slide_name]['set'] == mode):
            TP, FP, FN = 0,0,0
#             print(i)
            y_true =  np.array(metadata[slide_name]['bbox'])
            y_pred = np.array(metadata[slide_name]['pred_bbox'])

            if(len(y_true) == 0):continue
            # if(len(y_pred) == 0):continue

            scores = y_pred[:, 4] if len(y_pred != 0) else []
#             y_pred = y_pred[:, :4]
            
            to_keep = scores>det_tresh
            y_pred = y_pred[to_keep]
            
            scores = y_pred[:, 4] if len(y_pred != 0) else []
            y_pred = y_pred[:, :4] if len(y_pred != 0) else []
            
            y_pred, _ = non_max_suppression_by_distance(y_pred, scores, 17.5, max(det_tresh, 0.05))
            # y_pred, _ = nms(y_pred, scores,  0.3)
            # print(y_pred.shape)

            center_true_x = (y_true[:,0] + y_true[:, 2]) / 2
            center_true_y = (y_true[:,1] + y_true[:, 3]) / 2
            center_x = (y_pred[:,0] + y_pred[:, 2]) / 2
            center_y = (y_pred[:,1] + y_pred[:, 3]) / 2

            centers_DB = np.concatenate([center_true_x[:, None], center_true_y[:, None]], axis = 1)

            isDet = np.zeros(y_pred.shape[0]+centers_DB.shape[0])
            isDet[0:y_pred.shape[0]]=1 # mark as detection, rest ist GT

            if (centers_DB.shape[0]>0):
                center_x = np.hstack((center_x, centers_DB[:,0]))
                center_y = np.hstack((center_y, centers_DB[:,1]))
            radius=25


            # set up kdtree 
            X = np.dstack((center_x, center_y))[0]


            try:
                tree = KDTree(X)
            except:
                print('Shapes of X: ',X.shape)
                raise Error()

            ind = tree.query_radius(X, r=radius)
            annotationWasDetected = {x: 0 for x in np.where(isDet==0)[0]}
            DetectionMatchesAnnotation = {x: 0 for x in np.where(isDet==1)[0]}

            # check: already used results
            alreadyused=[]
            for i in ind:
                if len(i)==0:
                    continue
                if np.any(isDet[i]) and np.any(isDet[i]==0):
                    # at least 1 detection and 1 non-detection --> count all as hits
                    for j in range(len(i)):
                        if not isDet[i][j]: # is annotation, that was detected
                            if i[j] not in annotationWasDetected:
                                print('Missing key ',j, 'in annotationWasDetected')
                                raise ValueError('Ijks')
                            annotationWasDetected[i[j]] = 1
                        else:
                            if i[j] not in DetectionMatchesAnnotation:
                                print('Missing key ',j, 'in DetectionMatchesAnnotation')
                                raise ValueError('Ijks')

                            DetectionMatchesAnnotation[i[j]] = 1

            TP = np.sum([annotationWasDetected[x]==1 for x in annotationWasDetected.keys()])
            FN = np.sum([annotationWasDetected[x]==0 for x in annotationWasDetected.keys()])

            FP = np.sum([DetectionMatchesAnnotation[x]==0 for x in DetectionMatchesAnnotation.keys()])
            sum_TP += TP
            sum_FP += FP 
            sum_FN += FN
            
#             print(len(annotationWasDetected), slide_name)
            
            dummy = []
            metadata[slide_name]['GT'] = [DetectionMatchesAnnotation[x] for x in DetectionMatchesAnnotation]
    return sum_TP, sum_FP, sum_FN, metadata
    
def query(meta, thresh, mode = 'test'):
    sum_TP, sum_FP, sum_FN, DetectionMatchesAnnotation = getf1(meta,  thresh, mode = mode)
    F1 = 100 * 2*sum_TP / (2*sum_TP + sum_FP + sum_FN)
    print( "At thresh = {:.2f} TP = {}, FN = {}, FP = {}, F1 = {:.2f}".format(thresh, sum_TP, sum_FN, sum_FP, F1))
    return DetectionMatchesAnnotation, (thresh, sum_TP, sum_FN, sum_FP, F1)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ip", "--pkl_path", required=True, help="pkl metadata path")
args = vars(parser.parse_args())

# eval_path = "tools/result/ambi_test.pkl"
eval_path = args["pkl_path"]
get_raw_result = False

metadata = pickle.load( open( eval_path, "rb" ) )
if(not get_raw_result):
    max_F1 = -1
    optimal_thresh = None
    for i in np.arange(1, 0.05, -0.01):
        try:
            _, x = query(metadata.copy(), i)  
            thresh, sum_TP, sum_FN, sum_FP, F1 = x
            recall = sum_TP/ (sum_TP + sum_FN)
            precision = sum_TP/ (sum_TP + sum_FP)
            metric = F1
            if(metric > max_F1):
                max_F1 = metric
                optimal_thresh = i
        except:
            precision = 0
            recall = 0
        print("precision = {:.3f}, recall = {:.3f}".format(precision, recall) )
    print("Best F1 = {:.2f} at thresh = {:.2f}".format(max_F1, optimal_thresh))
else:
    detection_result = query(metadata.copy(), 0, mode = 'train')
    pickle.dump(detection_result, open(eval_path.split('.')[0] + '_eval.pkl', 'wb') )
