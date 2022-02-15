''' 

WSI download code from https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/Setup.ipynb, and
https://github.com/DeepPathology/MITOS_WSI_CCMCT/blob/master/Setup.ipynb. 
'''
import urllib.request
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Dataset Selection")
args = vars(parser.parse_args())
dataset = args["dataset"]

if(dataset == 'CCMCT'):
    WSI_path = 'data/dataset/CCMCT/WSI'
    downloadableFiles = {'databases/MITOS_WSI_CCMCT_Tumorzone.sqlite': 
                          'https://ndownloader.figshare.com/files/16261586?private_link=a82ddb634864c24f4aee' ,
                     'databases/MITOS_WSI_CCMCT_ODAEL.sqlite':
                          'https://ndownloader.figshare.com/files/16261571?private_link=a82ddb634864c24f4aee' ,
                     'databases/MITOS_WSI_CCMCT_HEAEL.sqlite':
                          'https://ndownloader.figshare.com/files/16261583?private_link=a82ddb634864c24f4aee' ,
                     'databases/MITOS_WSI_CCMCT_MEL.sqlite':
                          'https://ndownloader.figshare.com/files/16261574?private_link=a82ddb634864c24f4aee' ,
                     'WSI/96274538c93980aad8d6.svs': # 3
                          'https://ndownloader.figshare.com/files/16261559?private_link=a82ddb634864c24f4aee', 
                     'WSI/1018715d369dd0df2fc0.svs': # 20
                          'https://ndownloader.figshare.com/files/16261562?private_link=a82ddb634864c24f4aee',
                     'WSI/9374efe6ac06388cc877.svs': # 26
                          'https://ndownloader.figshare.com/files/16261553?private_link=a82ddb634864c24f4aee',
                     'WSI/552c51bfb88fd3e65ffe.svs': # 27
                          'https://ndownloader.figshare.com/files/16261556?private_link=a82ddb634864c24f4aee',
                     'WSI/285f74bb6be025a676b6.svs': # 29
                          'https://ndownloader.figshare.com/files/16261550?private_link=a82ddb634864c24f4aee',
                     'WSI/91a8e57ea1f9cb0aeb63.svs': # 24
                          'https://ndownloader.figshare.com/files/16261544?private_link=a82ddb634864c24f4aee',
                     'WSI/70ed18cd5f806cf396f0.svs': # 35
                          'https://ndownloader.figshare.com/files/16261541?private_link=a82ddb634864c24f4aee',
                     'WSI/066c94c4c161224077a9.svs': # 25 
                          'https://ndownloader.figshare.com/files/16261547?private_link=a82ddb634864c24f4aee',
                     'WSI/39ecf7f94ed96824405d.svs': # 19
                          'https://ndownloader.figshare.com/files/16261529?private_link=a82ddb634864c24f4aee',
                     'WSI/34eb28ce68c1106b2bac.svs': # 14
                          'https://ndownloader.figshare.com/files/16261538?private_link=a82ddb634864c24f4aee',
                     'WSI/20c0753af38303691b27.svs': # 21
                          'https://ndownloader.figshare.com/files/16261532?private_link=a82ddb634864c24f4aee',
                     'WSI/3f2e034c75840cb901e6.svs': # 15
                          'https://ndownloader.figshare.com/files/16261505?private_link=a82ddb634864c24f4aee',
                     'WSI/2efb541724b5c017c503.svs': #22 
                          'https://ndownloader.figshare.com/files/16261520?private_link=a82ddb634864c24f4aee',
                     'WSI/2f2591b840e83a4b4358.svs':#23
                          'https://ndownloader.figshare.com/files/16261514?private_link=a82ddb634864c24f4aee',
                     'WSI/8bebdd1f04140ed89426.svs': # 17
                          'https://ndownloader.figshare.com/files/16261523?private_link=a82ddb634864c24f4aee',
                     'WSI/8c9f9618fcaca747b7c3.svs': # 9
                          'https://ndownloader.figshare.com/files/16261526?private_link=a82ddb634864c24f4aee',
                     'WSI/2f17d43b3f9e7dacf24c.svs': # 8
                          'https://ndownloader.figshare.com/files/16261535?private_link=a82ddb634864c24f4aee',
                     'WSI/f3741e764d39ccc4d114.svs': # 31
                          'https://ndownloader.figshare.com/files/16261493?private_link=a82ddb634864c24f4aee',
                     'WSI/fff27b79894fe0157b08.svs': # 7
                          'https://ndownloader.figshare.com/files/16261490?private_link=a82ddb634864c24f4aee',
                     'WSI/f26e9fcef24609b988be.svs': # 6
                          'https://ndownloader.figshare.com/files/16261496?private_link=a82ddb634864c24f4aee',
                     'WSI/dd4246ab756f6479c841.svs': # 18
                          'https://ndownloader.figshare.com/files/16261487?private_link=a82ddb634864c24f4aee',
                     'WSI/c3eb4b8382b470dd63a9.svs': # 4
                          'https://ndownloader.figshare.com/files/16261466?private_link=a82ddb634864c24f4aee',
                     'WSI/c86cd41f96331adf3856.svs': # 30
                          'https://ndownloader.figshare.com/files/16261475?private_link=a82ddb634864c24f4aee',
                     'WSI/c91a842257ed2add5134.svs': # 1
                          'https://ndownloader.figshare.com/files/16261481?private_link=a82ddb634864c24f4aee',
                     'WSI/dd6dd0d54b81ebc59c77.svs': # 28
                          'https://ndownloader.figshare.com/files/16261478?private_link=a82ddb634864c24f4aee',
                     'WSI/be10fa37ad6e88e1f406.svs': # 11
                          'https://ndownloader.figshare.com/files/16261469?private_link=a82ddb634864c24f4aee',
                     'WSI/ce949341ba99845813ac.svs': # 34
                          'https://ndownloader.figshare.com/files/16261484?private_link=a82ddb634864c24f4aee',
                     'WSI/a0c8b612fe0655eab3ce.svs': # 13
                          'https://ndownloader.figshare.com/files/16261424?private_link=a82ddb634864c24f4aee',
                     'WSI/add0a9bbc53d1d9bac4c.svs': # 2
                          'https://ndownloader.figshare.com/files/16261436?private_link=a82ddb634864c24f4aee',
                     'WSI/2e611073cff18d503cea.svs': # 32
                          'https://ndownloader.figshare.com/files/16261439?private_link=a82ddb634864c24f4aee',
                     'WSI/0e56fd11a762be0983f0.svs': # 31
                          'https://ndownloader.figshare.com/files/16261442?private_link=a82ddb634864c24f4aee',
                     'WSI/ac1168b2c893d2acad38.svs': # 12
                          'https://ndownloader.figshare.com/files/16261445?private_link=a82ddb634864c24f4aee',
                    }

elif(dataset == 'CMC'):
    WSI_path = 'data/dataset/CMC/WSI'

    downloadableFiles = {'WSI/deb768e5efb9d1dcbc13.svs' : #18
                            'https://ndownloader.figshare.com/files/22407414?private_link=be072bf30fd3f63b03cc',
                        'WSI/d37ab62158945f22deed.svs' : #19
                            'https://ndownloader.figshare.com/files/22585835?private_link=be072bf30fd3f63b03cc',
                        'WSI/022857018aa597374b6c.svs': #1,
                            'https://ndownloader.figshare.com/files/22407537?private_link=be072bf30fd3f63b03cc',
                        'WSI/69a02453620ade0edefd.svs': #2
                            'https://ndownloader.figshare.com/files/22407411?private_link=be072bf30fd3f63b03cc', 
                        'WSI/a8773be388e12df89edd.svs': #3
                            'https://ndownloader.figshare.com/files/22407540?private_link=be072bf30fd3f63b03cc',
                        'WSI/c4b95da36e32993289cb.svs': #4
                            'https://ndownloader.figshare.com/files/22407552?private_link=be072bf30fd3f63b03cc',
                        'WSI/3d3d04eca056556b0b26.svs': #5
                            'https://ndownloader.figshare.com/files/22407585?private_link=be072bf30fd3f63b03cc',
                        'WSI/d0423ef9a648bb66a763.svs': #6
                            'https://ndownloader.figshare.com/files/22407624?private_link=be072bf30fd3f63b03cc',
                        'WSI/50cf88e9a33df0c0c8f9.svs': #7
                            'https://ndownloader.figshare.com/files/22407531?private_link=be072bf30fd3f63b03cc',
                        'WSI/084383c18b9060880e82.svs': #8
                            'https://ndownloader.figshare.com/files/22407486?private_link=be072bf30fd3f63b03cc',
                        'WSI/4eee7b944ad5e46c60ce.svs': #9
                            'https://ndownloader.figshare.com/files/22407528?private_link=be072bf30fd3f63b03cc',
                        'WSI/2191a7aa287ce1d5dbc0.svs' : #10
                            'https://ndownloader.figshare.com/files/22407525?private_link=be072bf30fd3f63b03cc',
                        'WSI/13528f1921d4f1f15511.svs' : #11
                            'https://ndownloader.figshare.com/files/22407519?private_link=be072bf30fd3f63b03cc',
                        'WSI/2d56d1902ca533a5b509.svs' : #12
                            'https://ndownloader.figshare.com/files/22407522?private_link=be072bf30fd3f63b03cc',
                        'WSI/460906c0b1fe17ea5354.svs' : #13
                            'https://ndownloader.figshare.com/files/22407447?private_link=be072bf30fd3f63b03cc',
                        'WSI/da18e7b9846e9d38034c.svs' : #14
                            'https://ndownloader.figshare.com/files/22407453?private_link=be072bf30fd3f63b03cc',
                        'WSI/72c93e042d0171a61012.svs' : #15
                            'https://ndownloader.figshare.com/files/22407456?private_link=be072bf30fd3f63b03cc',
                        'WSI/b1bdee8e5e3372174619.svs' : #16
                            'https://ndownloader.figshare.com/files/22407423?private_link=be072bf30fd3f63b03cc',
                        'WSI/fa4959e484beec77543b.svs' : #17
                            'https://ndownloader.figshare.com/files/22407459?private_link=be072bf30fd3f63b03cc',
                        'WSI/e09512d530d933e436d5.svs' : #20
                            'https://ndownloader.figshare.com/files/22407465?private_link=be072bf30fd3f63b03cc',
                        'WSI/d7a8af121d7d4f3fbf01.svs' : #21
                            'https://ndownloader.figshare.com/files/22407477?private_link=be072bf30fd3f63b03cc',
                        }
else:
    raise Exception("Only CMC and CCMCT datasets are supported.")


# Create folder for WSI if nonexistant
try:os.makedirs(WSI_path)
except: pass

tqdm.write('Downloading all files from figshare - take a coffee and sit down, this will take some while, we are downloading above 37GB ...')
    

from time import sleep    
sleep(0.5)

import requests

                    
for fname in tqdm(list(downloadableFiles.keys())):
    urllib.request.urlretrieve(downloadableFiles[fname],fname)


https://github.com/DeepPathology/MITOS_WSI_CCMCT/raw/master/databases/MITOS_WSI_CCMCT_ODAEL.sqlite