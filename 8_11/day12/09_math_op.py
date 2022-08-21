'''
张量的数学运算
'''
import tensorflow as tf

x = tf.constant([[1,2],[3,4]],dtype='float32')
y = tf.constant([[4,3],[2,1]],dtype='float32')

add = tf.add(x,y)
mul = tf.matmul(x,y)
log = tf.log(x)
reduce_sum = tf.reduce_sum(x,axis=1)
#计算张量片段和
data = tf.constant([1,2,3,4,5,6,7,8,9,10])
ids = tf.constant( [0,0,0,1,1,1,1,1,2,2])
segment = tf.segment_sum(data,ids)

with tf.Session() as sess:
    print(add.eval())
    print(mul.eval())
    print(log.eval())
    print(reduce_sum.eval())
    print(segment.eval())



