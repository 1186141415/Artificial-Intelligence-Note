'''
张量数据类型的转换
'''
import tensorflow as tf

ones = tf.ones(shape=[2,3],dtype='int64')
tensor1 = tf.constant([1.1,2.2,3.3],dtype='float32')

#转换类型
temp = tf.cast(ones,'float32')
# temp_str = tf.cast(tensor1,tf.string) # 报错

with tf.Session() as sess:
    print(temp.eval())
    # print(temp_str.eval())

