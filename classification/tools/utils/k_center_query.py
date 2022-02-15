import pickle
a = pickle.load(open('sampled_embedding.pkl', 'rb'))

import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

class KCenterGreedy():
    def __init__(self):
        super(KCenterGreedy, self)
        self.min_distances = None
        self.already_selected = []    
        self.features = None

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        if reset_dist: self.min_distances = None
        if only_new: cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
        if cluster_centers is not None:
          # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
                
    def query_batch(self, X, already_selected, N):
        self.features = X
        self.update_distances(already_selected, only_new=False, reset_dist=True)
        new_batch = []
        for _ in tqdm(range(N)):
            if self.already_selected is None: ind = np.random.choice(np.arange(self.n_obs))
            else: ind = np.argmax(self.min_distances)
            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        return new_batch

import numpy as np
def get_center(x):
    center_x = (x[:, 0] + x[:, 2])/2
    center_y = (x[:, 1] + x[:, 3])/2
    return np.concatenate([center_x[:, None], center_y[:, None]], axis = 1)
status = []
status2 = []
for i in a:
    if('pred_bbox1' in a[i]):
        
        y_true2 = np.array(a[i]['bbox'])
        true2_center_x = (y_true2[:,0] + y_true2[:, 2]) / 2
        true2_center_y = (y_true2[:,1] + y_true2[:, 3]) / 2
        true2_center = np.concatenate([true2_center_x[...,None], true2_center_y[..., None]], axis = 1)

        
        q1 =  np.array(np.array(a[i]['pred_bbox1'])[:, : -1])
        q2 = np.array(np.array(a[i]['pred_bbox1'])[:, -1])
        print(q2.shape)
        for j, k in zip(q1, q2):
    
            pred_center = np.array([int((j[0] + j[2])/2) , int((j[1] + j[3])/2)])[None, ]
            dist = pred_center - true2_center
            dist = np.sqrt(dist[:,0] ** 2 + dist[:, 1] ** 2)
            min_dist2 =  np.min(dist)
            cls = 'other'if min_dist2 > 50 else 'mitotic'
            if(cls != 'mitotic'):
                status.append([i, int((j[0] + j[2])/2) , int((j[1] + j[3])/2), cls])
                status2.append(k)
k_center = KCenterGreedy()
r = k_center.query_batch(np.array(status2), [0], 20000)
def show_im(x_center, y_center, slide):
    import matplotlib.pyplot as plt
    img = np.array(slide.read_region((x_center - 64, y_center - 64), 0, (128, 128)))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img[..., ::-1]
d = []
import cv2
base_path = '/home/schwan/Documents/NVIDIA/ReCasNet/detection/mmdetection/mmdetection/data/CMC/WSI/'
import openslide
for i in np.array(status)[np.array(r)]:
    slide = openslide.OpenSlide(base_path + i[0] )
    pred_center_x, pred_center_y =int(float(i[1])), int(float(i[2]))
    img = show_im(pred_center_x, pred_center_y , slide)
    
    file_name = 'sampled/{}/{}_{}_{}.png'.format(i[3].strip(), i[0], str(pred_center_x), str(pred_center_y))
    cv2.imwrite(file_name, img)
    d.append([i[0].split('.')[0], pred_center_y, int(float(i[2])), i[3], 'train'])
