import numpy as np
class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    2,
                'Tumorcells' :      3,
                'query' : 4
               }

test_class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    2,
                'Tumorcells' :      3,
                'query' : 4

}
img_size = (128, 128, 3)
NUM_CLASSES = 5
dataset_path =  '../detection/mmdetection/data/dataset/CCMCT/WSI/'
BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
train_mode = 'relocation_aux'
backbone = 'effnet'