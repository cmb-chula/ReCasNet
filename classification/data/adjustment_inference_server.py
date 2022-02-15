import threading
import time
import numpy as np
from collections import deque


class CenterAdjustmentInferenceServer(threading.Thread):
    def __init__(self, model, batch_size = 32, limit=False, max_queue_size=30, cls_weight = 1, seek_data = False, 
    get_embedding = False, pos_thresh = 0.2,  target=None, name=None):
        # super(DummyInferenceEngine, self).__init__()
        threading.Thread.__init__(self)
        self.target = target
        self.name = name
        self.model = model
        self.limit = limit
        self.max_queue_size = max_queue_size
        self.buffer = deque(maxlen=max_queue_size)
        self.prediction_result = []
        self.can_query_result = False
        self.cls_weight = cls_weight
        self.batch_size = batch_size
        self.batch_buffer = []
        self.loc_buffer = []
        self.seek_data = seek_data  
        self.get_embedding = get_embedding  
        self.pos_thresh = pos_thresh

    def push(self, X):
        """Push image into buffer (shared memory).
        # Arguments
            X: Arrray(H x W x 3), body image.
        # Returns
            None
        """
        # print(len(self.buffer))
        while(len(self.buffer) == self.max_queue_size):
            time.sleep(1e-6)
        self.buffer.append(X)

    def grab(self):
        """Grab image from the buffer (shared memory).
        # Arguments
            engine : InferenceEngine, engine used for inference
        # Returns
            X: Arrray(H x W x 3), body image.
        """

        while(len(self.buffer) == 0):
            time.sleep(1e-6)
        X = self.buffer.popleft()
        return X

    def get_prediction_result(self):
        while(self.can_query_result == False):
            time.sleep(1e-6)
        output = self.prediction_result.copy()
        #reset state
        self.can_query_result = False
        self.prediction_result = []
        return output

    def inference_batch(self):
        x = np.array(self.batch_buffer, dtype = np.float32)
        res = self.model(x)
        flag = False

        if(len(res) == 2 and len(res[0].shape) == 2):
            res2 = res[1]
            res = res[0]
            flag = True
        # print(flag)
        if(flag):
            for idx, (coors, dif_loc, obj_conf) in enumerate(zip(np.array(self.loc_buffer), res.numpy(), res2.numpy())):

                # print(dif_loc * 64)
                # self.prediction_result.append([xmin  + (dif_loc[0] -0.5) * 64, ymin + (dif_loc[1] -0.5) * 64, xmax + (dif_loc[0] -0.5) * 64, ymax + (dif_loc[1] -0.5) * 64, conf])
                xmin, ymin, xmax, ymax, conf = coors
                # print(dif_loc, obj_conf)
                w = 0
                rescore = (1 - w) * conf + w * obj_conf[0]
                max_conf = np.max(obj_conf)
                if( obj_conf[0] > self.pos_thresh):
                # if( conf > 0.4):
                    if(not  self.seek_data):
                        # self.prediction_result.append([xmin, ymin, xmax, ymax, rescore])
                        self.prediction_result.append([xmin + dif_loc[0] * x.shape[1] / 2, ymin + dif_loc[1] * x.shape[1] / 2, xmax + dif_loc[0] * x.shape[1] / 2, ymax + dif_loc[1] * x.shape[1] / 2, rescore])
                    else:
                        self.prediction_result.append([xmin + dif_loc[0] * x.shape[1] / 2, ymin + dif_loc[1] * x.shape[1] / 2, xmax + dif_loc[0] * x.shape[1] / 2, ymax + dif_loc[1] * x.shape[1] / 2, (xmin, ymin, xmax, ymax)])
                else:
                    if(not  self.seek_data):
                        self.prediction_result.append([xmin, ymin, xmax, ymax, rescore])
                    else:
                        self.prediction_result.append([xmin, ymin, xmax, ymax, (xmin, ymin, xmax, ymax)])

        else:
            # print(res.numpy())   
            for idx, (coors, dif_loc) in enumerate(zip(np.array(self.loc_buffer), res.numpy())):
                # print(dif_loc * 64)
                # self.prediction_result.append([xmin  + (dif_loc[0] -0.5) * 64, ymin + (dif_loc[1] -0.5) * 64, xmax + (dif_loc[0] -0.5) * 64, ymax + (dif_loc[1] -0.5) * 64, conf])
                xmin, ymin, xmax, ymax, conf = coors
                if(not  self.seek_data):
                    self.prediction_result.append([xmin + dif_loc[0] * x.shape[1] / 2, ymin + dif_loc[1] * x.shape[1] / 2, xmax + dif_loc[0] * x.shape[1] / 2, ymax + dif_loc[1] * x.shape[1] / 2, conf])

                else:
                    if(self.get_embedding):
                        self.prediction_result.append([xmin, ymin, xmax , ymax , dif_loc])
                    else:
                        self.prediction_result.append([xmin + dif_loc[0] * x.shape[1] / 2, ymin + dif_loc[1] * x.shape[1] / 2, xmax + dif_loc[0] * x.shape[1] / 2, ymax + dif_loc[1] * x.shape[1] / 2, (xmin, ymin, xmax, ymax)])
        self.batch_buffer = []
        self.loc_buffer = []

    def run(self):
        """Start image loading thread.
        """
        while(True):
            x = self.grab()
            if(x is None): break
            xmin, ymin, xmax, ymax, conf, img = x
            # if(conf < 0.1 or conf > 0.9):
            #     self.prediction_result.append([xmin, ymin, xmax, ymax, conf ])
            #     continue
            self.batch_buffer.append(img)
            self.loc_buffer.append([xmin, ymin, xmax, ymax, conf])
            if(len(self.batch_buffer) == self.batch_size): 
                self.inference_batch()
        if(len(self.batch_buffer) > 0): self.inference_batch()
        self.prediction_result = self.prediction_result
        self.can_query_result = True


    def run_single_instance(self):
        """Start image loading thread.
        """
        while(True):
            x = self.grab()
            if(x is None): break
            xmin, ymin, xmax, ymax, conf, img = x
            # if(conf < 0.1 or conf > 0.9):
            #     self.prediction_result.append([xmin, ymin, xmax, ymax, conf ])
            #     continue
            res = self.model(np.array(img[None, ], dtype = np.float32))
            rescore_conf = (self.cls_weight) * res[0][0].numpy() + (1 - self.cls_weight) * conf
            self.prediction_result.append([xmin, ymin, xmax, ymax, rescore_conf])
        self.prediction_result = np.array(self.prediction_result, dtype = np.float32)
        self.can_query_result = True
