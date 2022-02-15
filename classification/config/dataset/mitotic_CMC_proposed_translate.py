import numpy as np
class_mapper = {
                'Mitosis'       :   0,        
                'Nonmitosis'   :   1,  
                'other' : 1
               }

test_class_mapper = {
                'Mitosis'       :   0,        
                'Nonmitosis'   :   1,    
                'other' : 1

}


img_size = (128, 128, 3)
NUM_CLASSES = 2
dataset_path = '../detection/mmdetection/mmdetection/data/CMC/WSI/'
BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
train_mode = 'relocation'
backbone = 'effnet'