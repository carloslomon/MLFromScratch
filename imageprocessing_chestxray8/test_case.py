import numpy as np
#from keras import backend as K
import tensorflow as tf

### ex3
def get_weighted_loss_test_case(sess):
    with sess.as_default() as sess:
        y_true = tf.keras.backend.constant(np.array(
            [[1, 1, 1],
             [1, 1, 0],
             [0, 1, 0],
             [1, 0, 1]]
        ))
        
        w_p = np.array([0.25, 0.25, 0.5])
        w_n = np.array([0.75, 0.75, 0.5])
        
        y_pred_1 = tf.keras.backend.constant(0.7*np.ones(y_true.shape))
        y_pred_2 = tf.keras.backend.constant(0.3*np.ones(y_true.shape))
    
    return y_true.numpy(), w_p, w_n, y_pred_1.numpy(), y_pred_2.numpy()