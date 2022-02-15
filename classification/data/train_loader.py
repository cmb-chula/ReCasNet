import time
import numpy as np
from .base_loader import BaseLoader
from sklearn.utils import shuffle
import numpy as np
import imgaug.augmenters as iaa
import tensorflow as tf
import scipy

class TrainLoader(BaseLoader):
    def __init__(self, target_size=(240, 96)):
        super().__init__(target_size)
    def actual_aug(self, img):
        import cv2
        if(np.random.rand() > 0.5):
            img = img[::-1, :, :]
        if(np.random.rand() > 0.5):
            img = img[:, :: -1, :]
        if(np.random.rand() > 0.5): img *= np.random.uniform(0.8, 1.2)
        img = np.clip(img, 0, 255)
        img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(-90, 90), row_axis=0, col_axis=1, channel_axis=2)

        if(np.random.rand() > 0.25):
            m = np.random.randint(3)
            if(m == 1):
                img = cv2.GaussianBlur(img, (5,5),cv2.BORDER_DEFAULT)
            if(m == 2):
                img = cv2.GaussianBlur(img, (3, 3),cv2.BORDER_DEFAULT)
        noise = 5 * np.random.randn(img.shape[0], img.shape[1], img.shape[2]) 
        img += noise
        img = np.clip(img, 0, 255)
        return img

    def aug(self, X):
        from multiprocessing.dummy import Pool
        p = Pool(8)
        aug_data = p.map(self.actual_aug, X)#[self.actual_aug(x) for x in X]
        p.close()
        return aug_data
    def augment_batch(self, X, Y):

        X = self.aug(X)
        X = np.array(X)
        alpha = 0.01
        Y[Y !=  1] = alpha / (Y.shape[1] - 1)
        Y[Y == 1] = 1 - alpha
        return X, Y

    @staticmethod
    def fetch_data_into_mem(generator, limit = None):
        """Load body images and label from the generator into a memory
        # Arguments
            generator : DataEngine, DataEngine generator and its subclass
        # Returns
            Array(nb. data x H x W x 3), all body images yield from the genertor .
            Array(nb. data x num_class), all body images yield from the genertor.
        """

        X, Y = [], []
        while(True):
            try:
                data = next(generator)
                
                x, y = data[0], data[1]
                X += list(x)
                Y += list(y)
                if(limit is not None and len(X) > limit):
                    break
                
            except (StopIteration, TypeError):
                break
        return np.array(X), np.array(Y)

    def query_label(self, file_path, training=False):

        classes = file_path.split('/')[0]
        one_hot = np.zeros(self.cfg.NUM_CLASSES, dtype = np.float32)
        if(training):
            one_hot[self.cfg.class_mapper[classes]] = 1
        else:
            one_hot[self.cfg.test_class_mapper[classes]] = 1
        return one_hot

    def run(self, cfg, data_path, batch_size=128, training=False, augment=False, get_label = True, return_index = False):

        self.cfg = cfg
        Y = None
        index_array = np.arange(len(data_path))

        while(True):
            batches = self.make_batches(len(data_path), batch_size)
            if(training):
               index_array =  shuffle(index_array, random_state=42)


            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]

                from multiprocessing.dummy import Pool
                p = Pool(4)
                X = p.map(self.load_image, [
                          cfg.dataset_path + '/' + data_path[x] for x in batch_ids])
                X = np.array(X)
                if(get_label):
                    Y = np.array([self.query_label(data_path[y], training)
                                for y in batch_ids], dtype = np.float32)
                else: 
                    Y = np.array([data_path[y] for y in batch_ids])

                p.close()
                if(augment and get_label):
                    X, Y = self.augment_batch(X, Y)
                X = np.array(X) 
                if(cfg.return_index):
                    yield X, Y, [cfg.dataset_path + '/' + data_path[x] for x in batch_ids]
                else:
                    yield X, Y

            if(not training):
                yield None
