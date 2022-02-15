# Export Dataset (patch-wise) to learn classifier easier


import numpy as np 
# import SlideRunner.general.dependencies
from SlideRunner.dataAccess.database import Database
from SlideRunner.dataAccess.annotations import ViewingProfile
import os
import openslide
import sqlite3
import cv2
import sys

DB = Database()

basepath='/home/schwan/Documents/NVIDIA/detection/mmdetection/mmdetection/data/mitotic/WSI/'
patchSize = 128


def listOfSlides(DB):
    DB.execute('SELECT uid,filename from Slides')
    return DB.fetchall()
test_slide_filenames = ['be10fa37ad6e88e1f406.svs',
                        'f3741e764d39ccc4d114.svs',
                        'c86cd41f96331adf3856.svs',
                        '552c51bfb88fd3e65ffe.svs',
                        '8c9f9618fcaca747b7c3.svs',
                        'c91a842257ed2add5134.svs',
                        'dd4246ab756f6479c841.svs',
                        'f26e9fcef24609b988be.svs',
                        '96274538c93980aad8d6.svs',
                        'add0a9bbc53d1d9bac4c.svs',
                        '1018715d369dd0df2fc0.svs']


DB.open('tools/SlideRunner/databases/MITOS_WSI_CCMCT_ODAEL.sqlite')

# ans_list = []
# f = open('cell_list.txt', "a")
data_list = []

for slide,filename in listOfSlides(DB):
    DB.loadIntoMemory(slide)
    # slide=openslide.open_slide(basepath+filename)

    for k in DB.annotations.keys():

        anno = DB.annotations[k]

        coord_x = anno.x1
        coord_y = anno.y1

        lu_x = int(coord_x - int(patchSize/2))
        lu_y = int(coord_y - int(patchSize/2))
        is_test = 'train' if filename not in test_slide_filenames else 'test'

        cell_class_list = [None, 'Granulocytes', 'Mitosis', 'Tumorcells', 'ambiguous', None, None, 'Mitosislike']
        cell_class = cell_class_list[anno.agreedClass]

        slide_name =  filename.split('.')[0]
        if(cell_class is None):continue
        if(cell_class == 'ambiguous' and is_test == 'test'):continue

        data_list.append([slide_name, lu_x, lu_y, cell_class, is_test])


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
import pandas as pd
df = pd.DataFrame(data_list, columns = ['name', 'top_x', 'top_y', 'class', 'set'])
df.to_csv('mitotic_data_ambi.csv', index = False)
# print(len(ans_list))
# import pickle
# pickle.dump(ans_list, open('cell_list.pkl', 'wb'))

