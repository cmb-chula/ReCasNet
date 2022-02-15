import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.utils import  plot_confusion_matrix, plot_to_image
import tensorflow.keras.backend as K
import pickle
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
        optimizer = tf.keras.optimizers.Adam(5e-4)

        # from tensorflow.keras import mixed_precision
        # optimizer = mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    if(loss_fn is None):
        loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    print(loss_fn)

    train_stat = {}

    max_val_acc = 0
    for step in tqdm(range(scheduler.NUM_ITERATION)):
        scheduler.schedule(step, optimizer)

        if(cfg.return_index):
            X_train, Y_train, indices = train_loader.grab()
        else:
            X_train, Y_train = train_loader.grab()
        # X_train = np.array(X_train, dtype=np.float32)
        # print(X_train.shape, Y_train.shape)
        loss = train_step(X_train, Y_train, model, loss_fn, optimizer)

        loss_np = np.array(loss.numpy(), dtype = np.float32)
        # for keys, values in zip(indices, loss_np):
        #     if(keys not in train_stat): train_stat[keys] = []
        #     train_stat[keys].append(values)


        if tf.equal(optimizer.iterations % 25, 0):
            tf.summary.scalar('loss/train_loss', K.mean(loss), step=optimizer.iterations)
            tf.summary.scalar('lr', optimizer.lr, step=optimizer.iterations)


        if tf.equal(optimizer.iterations % scheduler.val_freq, 0):
            y_true, y_pred = [], []
            while(True):
                val_data = val_loader.grab()
                if(val_data is None): break
                if(cfg.return_index): 
                    X_val, Y_val, _ = val_data
                else: 
                    X_val, Y_val = val_data
                y_true += list(Y_val)
                y_pred += list(test_step(X_val, Y_val, model, loss_fn, optimizer))
            val_loss = loss_fn(np.array(y_true, dtype = np.float32), np.array(y_pred, dtype = np.float32))

            y_true = np.argmax(np.array(y_true), axis = 1)
            y_pred = np.argmax(np.array(y_pred), axis = 1)

            val_accuracy = accuracy_score(y_true, y_pred)
            print(np.array(val_loss).mean())
            metric = val_accuracy
            # metric = np.array(val_loss).mean()
            if(metric > max_val_acc and optimizer.iterations > 2000):
                print("-->", metric)
                max_val_acc = metric
                model.save('{}.h5'.format(ckpt_path))

            tf.summary.image("Confusion Matrix", plot_to_image(plot_confusion_matrix(
                confusion_matrix(y_true, y_pred), cfg.test_class_mapper.keys())), step=optimizer.iterations)
            tf.summary.scalar('loss/val_loss', K.mean(val_loss), step=optimizer.iterations)
            tf.summary.scalar('accuracy/val_acc', val_accuracy, step=optimizer.iterations)

    # pickle.dump(train_stat, open('train_stat_step2.pkl', 'wb'))
    model.save('{}_final.h5'.format(ckpt_path))
    del train_loader
