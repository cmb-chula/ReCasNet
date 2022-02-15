# Export Dataset (patch-wise) to learn classifier easier


import numpy as np 
# import SlideRunner.general.dependencies
from SlideRunner.dataAccess.database import Database
from SlideRunner.dataAccess.annotations import *
import os
import openslide
import sqlite3
import cv2
import sys
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Dataset Selection")
args = vars(parser.parse_args())
dataset = args["dataset"]

patchSize = 64

if(dataset == 'CCMCT'): 
    WSI_path = 'data/dataset/CCMCT/WSI'
    cell_class_list = [None, 'Granulocytes', 'Mitosis', 'Tumorcells', None, None, None, 'Mitosislike']
    DB_path = 'data/database/MITOS_WSI_CCMCT_ODAEL.sqlite'
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
elif(dataset == 'CMC'): 
    WSI_path = 'data/dataset/CMC/WSI'
    cell_class_list = [None, 'Nonmitosis', 'Mitosis']
    DB_path = 'data/database/MITOS_WSI_CMC_CODAEL_TR_ROI.sqlite'
    test_slide_filenames = [
     '2d56d1902ca533a5b509.svs', 
     '69a02453620ade0edefd.svs', 
     '4eee7b944ad5e46c60ce.svs', 
     'b1bdee8e5e3372174619.svs', 
     'e09512d530d933e436d5.svs', 
     '022857018aa597374b6c.svs', 
     '13528f1921d4f1f15511.svs'] 
else:
    raise Exception("Only CMC and CCMCT datasets are supported.")



def listOfSlides(DB):
    DB.execute('SELECT uid,filename from Slides')
    return DB.fetchall()

DB = Database()
DB.open(DB_path)
data_list = []

for slide,filename in tqdm(listOfSlides(DB)):
    DB.loadIntoMemory(slide)
    slide=openslide.open_slide(WSI_path + '/' +filename)

    for k in DB.annotations.keys():

        anno = DB.annotations[k]
        if anno.deleted or anno.annotationType != AnnotationType.SPOT:
            continue
        coord_x = int(anno.x1)
        coord_y = int(anno.y1)

        istest = 'train/' if filename not in test_slide_filenames else 'test/'
        if(istest == 'test/'): continue

        cell_class = cell_class_list[anno.agreedClass]
        if(cell_class == None): continue
        
        slide_name =  filename.split('.')[0]
        data_list.append([slide_name, coord_x, coord_y, cell_class, istest[:-1]])
        # result = slide_name + '_' + str(lu_x) + '_' + str(lu_y)
        # if(istest == 'train/'):
        #     f.write('{}\n'.format(result))
        #     ans_list.append(anno.agreedClass == 2)
        

        # # if(istest != 'test/'):
        # #     continue
        # img = np.array(slide.read_region(location=(lu_x, lu_y), level=0, size=(patchSize, patchSize)))
        # img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # if (anno.agreedClass==2):
        #     fname = ('DataODAEL/')+istest+'Mitosis/{}_{}_{}.png'.format(slide_name, coord_x, coord_y)
        #     if not cv2.imwrite(fname, img):
        #           print('Write failed: ',fname)

        # if (anno.agreedClass==7):
        #     cv2.imwrite(('DataODAEL/')+istest+'Mitosislike/{}_{}_{}.png'.format(slide_name, coord_x, coord_y), img)

        # if (anno.agreedClass==3):
        #     cv2.imwrite(('DataODAEL/')+istest+'Tumorcells/{}_{}_{}.png'.format(slide_name, coord_x, coord_y), img)

        # if (anno.agreedClass==1):
        #     cv2.imwrite(('DataODAEL/')+istest+'Granulocytes/{}_{}_{}.png'.format(slide_name, coord_x, coord_y), img) 
import pandas as pd
df = pd.DataFrame(data_list, columns = ['name', 'center_x',	'center_y', 'class', 'set'])
df.to_csv('data/database/' + dataset + '_trainlabel.csv', index = False)
