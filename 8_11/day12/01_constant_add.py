'''
快速开始，感受tensorflow
'''
import tensorflow as tf

#定义
const1 = tf.constant(100.0)
const2 = tf.constant(200.0)
res = tf.add(const1,const2)

#执行
with tf.Session() as sess:
    print(sess.run(res))



a = tf.constant(100.0)
var = tf.Variable(100.0)

