import numpy as np
class_mapper = {
                'Mitosis'       :   0,        
                'Nonmitosis'   :   1,  
                'kcenter' : 1

               }

test_class_mapper = {
                'Mitosis'       :   0,        
                'Nonmitosis'   :   1,    
                'kcenter' : 1

}

img_size = (128, 128, 3)
NUM_CLASSES = 2
dataset_path = 'data/dataset/labelled-pool/CMC/train_kcenter'
BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
return_index = False
train_mode = 'classification'
backbone = 'effnet'