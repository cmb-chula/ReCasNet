import sqlite3
import numpy as np
from SlideRunner.dataAccess.database import Database
from tqdm import tqdm
from pathlib import Path
import openslide
import time
from random import randint
from get_slide import get_slides
import pickle 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Dataset Selection")
args = vars(parser.parse_args())
dataset = args["dataset"]


size=512
if(dataset == 'CMC'):
    WSI_path = 'data/dataset/CMC/WSI'
    DB_path = 'data/database/MITOS_WSI_CMC_CODAEL_TR_ROI.sqlite'
elif(dataset == 'CCMCT'):
    WSI_path = 'data/dataset/CCMCT/WSI'
    DB_path = 'data/database/MITOS_WSI_CCMCT_ODAEL.sqlite'
else:
    raise Exception("Only CMC and CCMCT datasets are supported.")


database = Database()
database.open(DB_path)

test_slide_name = []
if(dataset == 'CMC'):
    test_slide_name = [
     '2d56d1902ca533a5b509.svs', 
     '69a02453620ade0edefd.svs', 
     '4eee7b944ad5e46c60ce.svs', 
     'b1bdee8e5e3372174619.svs', 
     'e09512d530d933e436d5.svs', 
     '022857018aa597374b6c.svs', 
     '13528f1921d4f1f15511.svs'] 
elif(dataset == 'CCMCT'):
    test_slide_name = [
    'be10fa37ad6e88e1f406.svs', 
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
else:
    pass

# if(dataset == 'CMC'):
#     test_slide = ['3', '10', '12', '15', '18', '21', '22']
# else:
#     pass

bboxs, slides,files = get_slides(database=database,basepath=WSI_path)
file_list = [i.split('/')[-1] for i in files]
print(len(file_list), slides)
c = 0
metadata = {}
for bbox, name in zip(bboxs, file_list):
    if(name not in metadata):
        metadata[name] = {}
    split = 'train' if name not in test_slide_name else 'test'
    metadata[name]['set'] = split
    metadata[name]['bbox'] =  np.array(bbox[0])
    print(len(bbox[0]), name, split)
    if(split == 'test'): c+=1
print(c)

pickle.dump(metadata, open('data/database/{}_label.pkl'.format(dataset), 'wb'))