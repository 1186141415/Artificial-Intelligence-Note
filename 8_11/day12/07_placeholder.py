'''
占位符
'''
import tensorflow as tf
import numpy as np

plhd = tf.placeholder('float32',shape=[None,3])

data = np.arange(1,7).reshape(2,3)

#运行
with tf.Session() as sess:
    print(sess.run(plhd,feed_dict={plhd:data}))

