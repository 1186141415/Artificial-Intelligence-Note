'''
创建多个图，指定Session运行
'''
import tensorflow as tf

const1 = tf.constant(100.0)
const2 = tf.constant(200.0)
res = tf.add(const1,const2)

#获取默认的图
graph = tf.get_default_graph()
print('默认的图:',graph)
#新建图
new_graph = tf.Graph()
with new_graph.as_default():
    new_op = tf.constant('hello world')

#运行
with tf.Session(graph=new_graph) as sess:
    # print(sess.run(res))
    print(sess.run(new_op))

with tf.Session(graph=graph) as sess1:
    print(sess1.run(res))

