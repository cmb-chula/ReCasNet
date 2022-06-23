import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.utils import  plot_confusion_matrix, plot_to_image
import tensorflow.keras.backend as K
@tf.function
def train_step(X, y_true, model, loss_fn, optimizer, reg_w):
    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        # Y_train = tf.convert_to_tensor(Y_train, dtype = tf.float32)
        loss_reg = loss_fn[0](y_true[0], y_pred[0])
        loss_reg *= y_true[1][:, 0] # first class is always positive <- IMPORTANT <--------------------
        loss_cls = loss_fn[1](y_true[1], y_pred[1])
        loss = reg_w * loss_reg +  ( 1 - reg_w) * loss_cls
    # loss *= np.array([5, 1, 1, 1, 1]).reshape(-1, 1)
    gradients = tape.gradient( loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss #* tf.convert_to_tensor(np.array([1.3, 1, 1, 1, 1]), dtype = tf.float32) 

@tf.function
def test_step(X, y_true, model, loss_fn, optimizer):
    y_pred = model(X, training=False)
    # loss = loss_fn(y_true, y_pred)
    return y_pred


def do_train(cfg, model, train_loader, val_loader, optimizer=None, scheduler=None, loss_fn=None, ckpt_path=None, reg_w = 0.9):
    if(optimizer is None):
        optimizer = tf.keras.optimizers.Adam(1e-3)
        # from tensorflow.keras import mixed_precision
        # optimizer = mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    print(reg_w)
    if(loss_fn is None):
        loss_fn = tf.keras.losses.MAE()
    print(loss_fn)

    best_val_loss = 100
    for step in tqdm(range(scheduler.NUM_ITERATION)):
        scheduler.schedule(step, optimizer)
        X_train, Y_train = train_loader.grab()
        loss = train_step(X_train, Y_train, model, loss_fn, optimizer, reg_w)

        if tf.equal(optimizer.iterations % 25, 0):
            tf.summary.scalar('loss/train_loss', K.mean(loss), step=optimizer.iterations)
            tf.summary.scalar('lr', optimizer.lr, step=optimizer.iterations)


        if tf.equal(optimizer.iterations % scheduler.val_freq, 0):
            y_true_cls, y_true_reg, y_pred_cls, y_pred_reg = [], [], [], []

            while(True):
                val_data = val_loader.grab()
                if(val_data is None): break
                X_val, Y_val = val_data
                y_true_reg += list(Y_val[0])
                y_true_cls += list(Y_val[1])
                
                y_pred = list(test_step(X_val, Y_val, model, loss_fn, optimizer))
                y_pred_reg += list(y_pred[0])
                y_pred_cls += list(y_pred[1])
                
            y_true_reg = np.array(y_true_reg, dtype = np.float32)
            y_true_cls = np.array(y_true_cls, dtype = np.float32)
            y_pred_reg = np.array(y_pred_reg, dtype = np.float32)
            y_pred_cls = np.array(y_pred_cls, dtype = np.float32)

            val_loss_reg = loss_fn[0]( y_true_reg, y_pred_reg )
            val_loss_cls = loss_fn[1]( y_true_cls, y_pred_cls )
        
            val_loss = reg_w * val_loss_cls + (1 -  reg_w) * tf.reduce_mean(val_loss_reg, axis = (1,2))

            metric = np.array(val_loss).mean()
            if(metric < best_val_loss and optimizer.iterations > 2000):
                # print("-->", metric)
                best_val_loss = metric
                model.save('{}.h5'.format(ckpt_path))

            tf.summary.scalar('loss/val_loss_reg', K.mean(val_loss_reg), step=optimizer.iterations)
            tf.summary.scalar('loss/val_loss_cls', K.mean(val_loss_cls), step=optimizer.iterations)
            tf.summary.scalar('loss/val_loss', K.mean(val_loss), step=optimizer.iterations)

    model.save('{}_final.h5'.format(ckpt_path))
    del train_loader
