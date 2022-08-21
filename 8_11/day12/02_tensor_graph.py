'''
查看默认的图，以及tensor的graph属性
'''
import tensorflow as tf

#定义
const1 = tf.constant(100.0)
const2 = tf.constant(200.0)
res = tf.add(const1,const2)
#获取默认的图
graph = tf.get_default_graph()
print('默认的图:',graph)

#运行
with tf.Session() as sess:
    print(sess.run([res,const1,const2]))
    print(const1.graph)
    print(const2.graph)
    print(res.graph)



