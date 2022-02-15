import pickle
train_pkl = pickle.load(open('../data/dataset/labelled-pool/test_embedding.pkl', 'rb'))
train_txt = open('../data/dataset/labelled-pool/test.txt', 'rb')

import matplotlib.pyplot as plt
import numpy as np
a = []
embedding_dict = {}
for pred, loc in zip(train_pkl, train_txt):
    embedding = pred[1][0]
    processed_loc = loc.decode().strip()
    svs_name, x_top, y_top = processed_loc.split('_')
    x_top = str(int(x_top))
    y_top = str(int(y_top))
    if(svs_name not in embedding_dict):
        embedding_dict[svs_name] = {}
    embedding_dict[svs_name][x_top + '_' + y_top] = embedding

    pickle.dump(embedding_dict, open('../data/dataset/labelled-pool/processed_test_embedding.pkl', 'wb'))