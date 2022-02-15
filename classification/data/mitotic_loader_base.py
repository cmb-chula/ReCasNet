import time
import numpy as np
from .base_loader import BaseLoader
from sklearn.utils import shuffle
import numpy as np
import imgaug.augmenters as iaa
import tensorflow as tf
import openslide


class MitoticLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def actual_aug(self, img):
        import cv2

        if(np.random.rand() > 0.5):
            img = img[::-1, :, :]
        if(np.random.rand() > 0.5):
            img = img[:, :: -1, :]
        if(np.random.rand() > 0.5): img *= np.random.uniform(0.8, 1.2)
        # img = tf.image.adjust_contrast( img, np.random.uniform(0.5, 1.5))
        img = np.clip(img, 0, 255)
        img = tf.keras.preprocessing.image.random_rotation(img,30, row_axis=0, col_axis=1, channel_axis=2)

        # img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(-90, 90), row_axis=0, col_axis=1, channel_axis=2)
        # img = tf.keras.preprocessing.image.random_zoom(img, (1, 1.1), row_axis=0, col_axis=1, channel_axis=2)

        if(np.random.rand() > 0.75):
            m = np.random.randint(3)
            if(m == 1):
                img = cv2.GaussianBlur(img, (5,5),cv2.BORDER_DEFAULT)
            if(m == 2):
                img = cv2.GaussianBlur(img, (3, 3),cv2.BORDER_DEFAULT)
        # noise = 10 * np.random.randn(img.shape[0], img.shape[1], img.shape[2]) 
        # img += noise
        # convert color from BGR to HSV
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # # # random hue
        # img[..., 0] += np.random.uniform(-18, 18)
        # img[..., 0][img[..., 0] > 360] -= 360
        # img[..., 0][img[..., 0] < 0] += 360
        # # convert color from HSV to BGR
        # img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        return img

    def aug(self, X):
        aug_data = [self.actual_aug(x) for x in X]
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


        r = np.random.rand(1)
        beta = 1
        cutmix_prob = 0.5
        if r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = np.random.permutation(len(X))
            
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(X.shape[2], X.shape[1], lam)
            X[:,  bby1:bby2, bbx1:bbx2, :] = X[rand_index, bby1:bby2, bbx1:bbx2, :]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.shape[1] * X.shape[2]))
            # compute output
            Y = lam.reshape(-1, 1) * Y + (1 - lam.reshape(-1, 1)) * Y[rand_index, :]
            Y = np.array(Y, dtype = np.float32)
        
        # alpha = 0.05
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

        slide_name, top_x, top_y, classes = img_path
        # if(training):
        #     top_x += np.random.randint(-5, 5)
        #     top_y += np.random.randint(-5, 5)
        # img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(-90, 90), row_axis=0, col_axis=1, channel_axis=2)

        if(training):
            x_center = top_x + 64
            y_center = top_y + 64
            # # img = np.array(self.slides[slide_name].read_region(location=(top_x, top_y), level=0, size=(128, 128)))
            # # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            # img = np.array(self.slides[slide_name].read_region(location=(x_center - 128, y_center - 128), level=0, size=(256, 256)))
            # # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            # # if(np.random.rand() > 0.75):
            # #     zoom_size = 8
            # #     img = img[zoom_size: 256-zoom_size, zoom_size: 256-zoom_size]
            # #     img = cv2.resize(img, (256, 256))

            # img = tf.keras.preprocessing.image.random_rotation(img,180, row_axis=0, col_axis=1, channel_axis=2)
            # img = img[64 : 256-64, 64 : 256-64, :]
            img = np.array(self.slides[slide_name].read_region(location=(top_x, top_y), level=0, size=(128, 128)))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

                # cv2.imwrite('a.png', img)
        else:
            img = np.array(self.slides[slide_name].read_region(location=(top_x, top_y), level=0, size=(128, 128)))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        # img = cv2.filter2D(img, -1, kernel)
        return img

    def run(self, cfg, data_path, batch_size=128, training=False, augment=False, get_label = True):

        self.cfg = cfg
        Y = None
        unique_slide = np.unique(data_path[:,0])
        self.slides = {slide : openslide.open_slide(cfg.dataset_path + slide + '.svs') for slide in unique_slide}

        while(True):
            if(training or get_label == False):
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
