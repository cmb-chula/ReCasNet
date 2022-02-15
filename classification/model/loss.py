import tensorflow as tf
def generalized_cross_entropy(y_true, y_pred):
    """
    2018 - nips - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    """
    q = tf.constant(0.7, dtype = tf.float32)
    t_loss = (tf.constant(1, dtype = tf.float32) - tf.pow(tf.reduce_sum(y_true * y_pred, axis=-1), q)) / q
    return tf.reduce_mean(t_loss)
