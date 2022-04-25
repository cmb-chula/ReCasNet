import time
import numpy as np
from .base_loader import BaseLoader
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import openslide
import cv2


class MitoticLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def actual_aug(self, img):

        if(np.random.rand() > 0.5):
            img = img[::-1, :, :]
        if(np.random.rand() > 0.5):
            img = img[:, :: -1, :]
        if(np.random.rand() > 0.25): img *= np.random.uniform(0.8, 1.2)

        img = np.clip(img, 0, 255)
        img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(-90, 90), row_axis=0, col_axis=1, channel_axis=2)
        # img = tf.keras.preprocessing.image.random_zoom(img, (1, 1.1), row_axis=0, col_axis=1, channel_axis=2)

        if(np.random.rand() > 0.25):
            m = np.random.randint(3)
            if(m == 1):
                img = cv2.GaussianBlur(img, (5,5),cv2.BORDER_DEFAULT)
            if(m == 2):
                img = cv2.GaussianBlur(img, (3, 3),cv2.BORDER_DEFAULT)

        return img

    def aug(self, X):
        from multiprocessing.dummy import Pool
        p = Pool(8)
        aug_data = p.map(self.actual_aug, X)#[self.actual_aug(x) for x in X]
        p.close()
        return aug_data
        
    def augment_batch(self, X, Y):
        """Augment training images. This method perform basic transformation and CutMix augmentation
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)
        # Arguments
            X : Array(batch size x H x W x 3), a tensor of body images
            Y : Array(batch size x num_class), a one hot label tensor of body images
        # Returns
            Array(batch size x H x W x 3), body image after augmented.
            Array(batch size x num_class), label after augmented.
        """

        X = self.aug(X)
        X = np.array(X)

        
        # alpha = 0.01
        # Y[Y !=  1] = alpha / (Y.shape[1] - 1)
        # Y[Y == 1] = 1 - alpha
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

        classes = file_path[-1]
        one_hot = np.zeros(self.cfg.NUM_CLASSES, dtype = np.float32)
        if(training):
            one_hot[self.cfg.class_mapper[classes]] = 1
        else:
            one_hot[self.cfg.test_class_mapper[classes]] = 1
        # print(one_hot, self.cfg.class_mapper[classes])
        return one_hot

    def load_image(self, img_path):
        """Returns an image give a file path.
        # Arguments
            img_path: String, path of an image.
            target_size: (Int, Int), the final size of the image (H, W).
        # Returns
            Array(H x W x 3) an read image from the given datapath. The image is in a RGB format.
        """
        import cv2
        training, img_path = img_path

        slide_name, x_center, y_center, classes = img_path

        # if(training):
        #     top_x += int(1.25 + 2 * 3.2 * np.random.randn())
        #     top_y += int(0.88 + 2 * 2.87* np.random.randn())
        # img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(-90, 90), row_axis=0, col_axis=1, channel_axis=2)
        top_x = x_center - 64
        top_y = y_center - 64

        if(training):
            img = np.array(self.slides[slide_name].read_region(location=(top_x, top_y), level=0, size=(128, 128)))
            # img = np.array(self.slides[slide_name].read_region(location=(x_center - 64, y_center - 64), level=0, size=(512, 512)))
            # img = tf.keras.preprocessing.image.random_rotation(img,90, row_axis=0, col_axis=1, channel_axis=2)
            # img = img[64 : 256-64, 64 : 256-64, :]

            # # if(np.random.rand() > 0.75):
            # #     zoom_size = 8
            # #     img = img[zoom_size: 256-zoom_size, zoom_size: 256-zoom_size]
            # #     img = cv2.resize(img, (256, 256))

            # img = np.array(self.slides[slide_name].read_region(location=(top_x, top_y), level=0, size=(128, 128)))

            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img = cv2.resize(img, (128, 128))
        else:
            img = np.array(self.slides[slide_name].read_region(location=(top_x, top_y), level=0, size=(128, 128)))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img = cv2.resize(img, (128, 128))
        return img

    def run(self, cfg, data_path, batch_size=128, training=False, augment=False, get_label = True):
        
        self.cfg = cfg
        Y = None
        unique_slide = np.unique(data_path[:,0])
        self.slides = {slide : openslide.open_slide(cfg.dataset_path + slide + '.svs') for slide in unique_slide}

        while(True):
            if(training):
                data_path = shuffle(data_path, random_state=42)
            batches = self.make_batches(len(data_path), batch_size)
            index_array = np.arange(len(data_path))
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                
                from multiprocessing.dummy import Pool
                p = Pool(16)
                X = p.map(self.load_image, [(training,data_path[x]) for x in batch_ids])
                X = np.array(X, dtype = np.float32)
                # print(X.shape)          
                if(get_label):
                    Y = np.array([self.query_label(data_path[y], training) for y in batch_ids], dtype = np.float32)
                else: 
                    Y = np.array([data_path[y] for y in batch_ids])

                p.close()
                if(augment and get_label):
                    X, Y = self.augment_batch(X, Y)
                X = np.array(X)
                yield X, Y

            if(not training):
                yield None
