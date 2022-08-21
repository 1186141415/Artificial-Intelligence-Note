'''
将图中的计算逻辑写入事件文件中，并在tensorboard中显示
'''
import tensorflow as tf

x = tf.constant(100.0,name='xxx')
y = tf.constant(200.0,name='yyy')
res = tf.add(x,y,name='add')

init_val = tf.random_normal(shape=[2,3])
var = tf.Variable(init_val,name='varvarvar')

#初始化
init_op = tf.global_variables_initializer()

#运行
with tf.Session() as sess:
    sess.run(init_op)
    fw = tf.summary.FileWriter('../summary/',graph=sess.graph)
    print(sess.run(res))













