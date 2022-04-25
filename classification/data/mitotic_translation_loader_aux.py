import time
import numpy as np
from .base_loader import BaseLoader
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import openslide


class MitoticTranslationLoaderAux(BaseLoader):
    def __init__(self):
        super().__init__()

    def actual_aug(self, data):
        import cv2
        img, label = data
        if(np.random.rand() > 0.5):
            img = img[::-1, :, :]
            label[1] *= -1
        if(np.random.rand() > 0.5):
            img = img[:, :: -1, :]
            label[0] *= -1

        if(np.random.rand() > 0.5): img *= np.random.uniform(0.8, 1.2)
        # img = tf.image.adjust_contrast( img, np.random.uniform(0.5, 1.5))
        img = np.clip(img, 0, 255)
        # img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(-90, 90), row_axis=0, col_axis=1, channel_axis=2)
        # img = tf.keras.preprocessing.image.random_zoom(img, (1, 1.1), row_axis=0, col_axis=1, channel_axis=2)

        if(np.random.rand() > 0.25):
            m = np.random.randint(3)
            if(m == 1):
                img = cv2.GaussianBlur(img, (5,5),cv2.BORDER_DEFAULT)
            if(m == 2):
                img = cv2.GaussianBlur(img, (3, 3),cv2.BORDER_DEFAULT)
        noise = 5 * np.random.randn(img.shape[0], img.shape[1], img.shape[2]) 
        img += noise
        img = np.clip(img, 0, 255)
        # convert color from BGR to HSV
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # # random hue
        # img[..., 0] += np.random.uniform(-18, 28)
        # img[..., 0][img[..., 0] > 360] -= 360
        # img[..., 0][img[..., 0] < 0] += 360

        # # convert color from HSV to BGR
        # img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return (img, label)

    def aug(self, X, Y):
        from multiprocessing.dummy import Pool
        p = Pool(8)
        aug_data = p.map(self.actual_aug, zip(X, Y))#[self.actual_aug(x) for x in X]
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

        aug_data = self.aug(X, Y)
        X = np.array([i[0] for i in aug_data], dtype = np.float32)
        Y = np.array([i[1] for i in aug_data], dtype = np.float32)
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
        interested_class = 1
        
        noise_x = int(6 * np.random.randn())
        noise_y = int(6 * np.random.randn())

        cls_onehot = np.zeros(self.cfg.NUM_CLASSES, dtype = np.float32)
        cls_onehot[self.cfg.class_mapper[classes]] = 1
        # print(img_path, cls_onehot)
        # if classes in ['Mitosis']:#, 'Mitosislike']:
        #     interested_class = 0
        # if(classes in ['other', 'UNK']):
        #     interested_class = 1
        # cls_onehot = np.zeros(2)
        # cls_onehot[interested_class] = 1
        # noise_x = np.random.randint(-12, 12)
        # noise_y = np.random.randint(-12, 12)

        x_center += noise_x
        y_center += noise_y

        # img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(-90, 90), row_axis=0, col_axis=1, channel_axis=2)
        patch_size = (self.cfg.img_size[0] // 2) 
        if(training):
            img = np.array(self.slides[slide_name].read_region(location=(x_center - patch_size, y_center - patch_size), level=0, size=(patch_size*2, patch_size*2)))
            # img = np.array(self.slides[slide_name].read_region(location=(x_center - 96, y_center - 96), level=0, size=(192, 192)))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
            # img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(-90, 90), row_axis=0, col_axis=1, channel_axis=2)
            # import scipy
            # angle = np.random.randint(-90, 90)
            # img = scipy.ndimage.rotate(img, angle, reshape = False)
            # new_noise_x = (noise_y* np.sin(angle *np.pi / 180) + noise_x * np.cos(angle *np.pi / 180))
            # new_noise_y = (noise_y* np.cos(angle *np.pi / 180) - noise_x * np.sin(angle *np.pi / 180))
            # noise_x, noise_y = new_noise_x, new_noise_y
            # img = img[32 : 192-32, 32 : 192-32, :]

        else:
            img = np.array(self.slides[slide_name].read_region(location=(x_center - patch_size, y_center - patch_size), level=0, size=(patch_size*2, patch_size*2)))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img = cv2.resize(img, (patch_size*2, patch_size*2))

        # if(interested_class == 0):
        #     noise_x, noise_y = 0, 0 
        return img, [noise_x, noise_y], cls_onehot

    def run(self, cfg, data_path, batch_size=128, training=False, augment=False, get_label = True):
        
        self.cfg = cfg
        Y = None
        unique_slide = np.unique(data_path[:,0])
        self.slides = {slide : openslide.open_slide(cfg.dataset_path + slide + '.svs') for slide in unique_slide}
        print(len(data_path))
        while(True):
            if(training):
                data_path = shuffle(data_path, random_state=42)
            batches = self.make_batches(len(data_path), batch_size)
            index_array = np.arange(len(data_path))
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                
                from multiprocessing.dummy import Pool
                p = Pool(4)
                XY = p.map(self.load_image, [(training, data_path[x]) for x in batch_ids])
                p.close()
                X = np.array([i[0] for i in XY], dtype = np.float32)
                Y = np.array([i[1] for i in XY], dtype = np.float32)
                Y_aux = np.array([i[2] for i in XY], dtype = np.float32)

                if(augment and get_label):
                    X, Y = self.augment_batch(X, Y)
                X = np.array(X)
                Y = - Y / (X.shape[1] / 2)
                # print(X.shape)
                #visual test
                # print(Y.max(), Y.min())
                # for idx, i in enumerate(zip(X, Y)):
                #     import cv2
                #     img, label = i
                #     print(img.shape, label)
                #     # print(img.shape, (label[0] + 64, label[1] + 64), (label[0] + 65, label[1] + 65), (0, 0, 255), 1)
                #     cv2.rectangle(img, (int(label[0] + 63), int(label[1] + 63)), (int(label[0] + 65), int(label[1] + 65)), (0, 0, 255), 2)
                #     cv2.imwrite('placeholder/{}.png'.format(idx), np.array(img, dtype = np.uint8))
                                
                yield X, [Y, Y_aux]

            if(not training):
                yield None
