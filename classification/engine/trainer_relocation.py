import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.utils import  plot_confusion_matrix, plot_to_image
import tensorflow.keras.backend as K
@tf.function
def train_step(X, y_true, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        # Y_train = tf.convert_to_tensor(Y_train, dtype = tf.float32)
        loss = loss_fn(y_true, y_pred)
    # loss *= np.array([5, 1, 1, 1, 1]).reshape(-1, 1)
    gradients = tape.gradient( loss, 
                    model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss #* tf.convert_to_tensor(np.array([1.3, 1, 1, 1, 1]), dtype = tf.float32) 

@tf.function
def test_step(X, y_true, model, loss_fn, optimizer):
    y_pred = model(X, training=False)
    # loss = loss_fn(y_true, y_pred)
    return y_pred


def do_train(cfg, model, train_loader, val_loader, optimizer=None, scheduler=None, loss_fn=None, ckpt_path=None):
    if(optimizer is None):
        optimizer = tf.keras.optimizers.Adam(1e-3)
        # from tensorflow.keras import mixed_precision
        # optimizer = mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    if(loss_fn is None):
        loss_fn = tf.keras.losses.MAE()
    print(loss_fn)


    best_val_loss = 100
    for step in tqdm(range(scheduler.NUM_ITERATION)):
        scheduler.schedule(step, optimizer)
        X_train, Y_train = train_loader.grab()
        loss = train_step(X_train, Y_train, model, loss_fn, optimizer)

        if tf.equal(optimizer.iterations % 25, 0):
            tf.summary.scalar('loss/train_loss', K.mean(loss), step=optimizer.iterations)
            tf.summary.scalar('lr', optimizer.lr, step=optimizer.iterations)


        if tf.equal(optimizer.iterations % scheduler.val_freq, 0):
            y_true, y_pred = [], []
            while(True):
                val_data = val_loader.grab()
                if(val_data is None): break
                X_val, Y_val = val_data
                y_true += list(Y_val)
                y_pred += list(test_step(X_val, Y_val, model, loss_fn, optimizer))
            val_loss = loss_fn(np.array(y_true, dtype = np.float32), np.array(y_pred, dtype = np.float32))

            print(np.array(val_loss).mean())
            metric = np.array(val_loss).mean()
            if(metric < best_val_loss and optimizer.iterations > 2000):
                print("-->", metric)
                best_val_loss = metric
                model.save('{}.h5'.format(ckpt_path))

            # tf.summary.image("Confusion Matrix", plot_to_image(plot_confusion_matrix(
            #     confusion_matrix(y_true, y_pred), cfg.test_class_mapper.keys())), step=optimizer.iterations)
            tf.summary.scalar('loss/val_loss', K.mean(val_loss), step=optimizer.iterations)
    model.save('{}_final.h5'.format(ckpt_path))
    del train_loader
