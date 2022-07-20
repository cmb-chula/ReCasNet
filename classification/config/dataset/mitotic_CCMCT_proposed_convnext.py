import numpy as np
class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    2,
                'Tumorcells' :      3,
                'Disagreement' : 3
               }

test_class_mapper = {
                'Mitosis'       :   0,        
                'Mitosislike'   :   1,          
                'Granulocytes' :    2,
                'Tumorcells' :      3,
                'Disagreement' : 3

}

img_size = (128, 128, 3)
NUM_CLASSES = 4
dataset_path = 'data/dataset/labelled-pool/CCMCT/train_disagreement'
BATCH_SIZE = 64
return_embedding = False
pretrain_path = None
return_index = True
train_mode = 'classification'
backbone = 'convnext'