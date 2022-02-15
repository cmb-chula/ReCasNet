# Export Dataset (patch-wise) to learn classifier easier


import numpy as np 
# import SlideRunner.general.dependencies
import os
import openslide
import sqlite3
import cv2
import sys


basepath= '../detection/mmdetection/mmdetection/data/mitotic/WSI/'
import pandas as pd
df = pd.read_csv('tools/obj_center_w_extra_sample.csv').values


patchSize = 128
data_list = []
from tqdm import tqdm
slides = {}
for filename in tqdm(df):
    name, coord_x, coord_y, classes, t_set = filename

    if(t_set == 'train'):
        if(name not in slides): 
            slides[name] =  openslide.open_slide(basepath + name + '.svs')
        slide = slides[name]
        lu_x = int(coord_x - int(patchSize/2))
        lu_y = int(coord_y - int(patchSize/2))
        img = np.array(slide.read_region(location=(lu_x, lu_y), level=0, size=(patchSize, patchSize)))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite('data/dataset/labelled-pool/DataODAEL/train_fix/{}/{}_{}_{}.png'.format(classes, name, int(coord_x), int(coord_y)), img)

        # result = slide_name + '_' + str(lu_x) + '_' + str(lu_y)
        # if(istest == 'train/'):
        #     f.write('{}\n'.format(result))
        #     ans_list.append(anno.agreedClass == 2)
        

        # # if(istest != 'test/'):
        # #     continue

        # if (anno.agreedClass==2):
        #     fname = ('DataODAEL/')+istest+'Mitosis/%d.png' % (k)
        #     if not cv2.imwrite(fname, img):
        #           print('Write failed: ',fname)

        # if (anno.agreedClass==7):
        #     cv2.imwrite(('DataODAEL/')+istest+'Mitosislike/%d.png' % k, img)

        # if (anno.agreedClass==3):
        #     cv2.imwrite(('DataODAEL/')+istest+'Tumorcells/%d.png' %k, img)

        # if (anno.agreedClass==1):
        #     cv2.imwrite(('DataODAEL/')+istest+'Granulocytes/%d.png' %k, img) 
