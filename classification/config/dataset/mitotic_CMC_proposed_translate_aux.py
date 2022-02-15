import numpy as np
class_mapper = {
                'Mitosis'       :   0,        
                'Nonmitosis'   :   1,  
                'certain_CMC'   :   2,       
                'query' : 2,   
               }

test_class_mapper = {
                'Mitosis'       :   0,        
                'Nonmitosis'   :   1,    
                'certain_CMC'   :   2,         
                'query' : 2,   
       
}

img_size = (128, 128, 3)
NUM_CLASSES = 3
dataset_path = '../detection/mmdetection/data/dataset/CMC/WSI/'
BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
train_mode = 'relocation_aux'
backbone = 'effnet'