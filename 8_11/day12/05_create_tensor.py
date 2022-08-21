'''
创建张量示例
'''
import tensorflow as tf
import numpy as np

const1 = tf.constant(100.0)
const2 = tf.constant([1,2,3,4,5,6])
const3 = tf.constant(np.arange(1,9).reshape(2,2,2))
zeros = tf.zeros(shape=[2,3],dtype='float32')
ones = tf.ones(shape=[2,3],dtype='float32')
zeros_like = tf.zeros_like(const3)
random = tf.random_normal(shape=[10],
                          mean=0.0,
                          stddev=1.0,
                          dtype='float32')

#运行
with tf.Session() as sess:
    print(const1.eval())
    print(const2.eval())
    print(const3.eval())
    print(zeros.eval())
    print(ones.eval())
    print(zeros_like.eval())
    print(random.eval())




