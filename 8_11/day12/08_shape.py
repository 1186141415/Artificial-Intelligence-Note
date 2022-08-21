'''
张量的形状改变
静态形状:初始状态，只能设置一次，并且固定下来不能设置
动态形状
'''
import tensorflow as tf

plhd = tf.placeholder('float32',shape=[None,3])
#设置静态形状
plhd.set_shape([4,3])
print(plhd)

#动态形状
new_plhd = tf.reshape(plhd,[1,4,3])
print(new_plhd)
new_plhd2 = tf.reshape(new_plhd,[1,1,4,3])
print(new_plhd2)






