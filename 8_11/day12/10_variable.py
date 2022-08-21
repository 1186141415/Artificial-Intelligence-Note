'''
变量
'''
import tensorflow as tf

init_val = tf.random_normal(shape=[2,3])

var = tf.Variable(init_val)

#初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op) #执行初始化
    print(sess.run(var))




