'''查看张量的属性'''
import tensorflow as tf

const1 = tf.constant([1,2,3,4,5,6,7],dtype='int64')

with tf.Session() as sess:
    print(sess.run(const1))
    print('name:',const1.name)
    print('dtype:',const1.dtype)
    print('shape:',const1.shape)
    print('op:',const1.op)
    print('graph:',const1.graph)