'''
简单的线性回归
'''
import tensorflow as tf
import os

#数据准备
x = tf.random_normal([100,1],mean=1.75,stddev=0.5,name='xxx')
#根据y=2x+5
y = tf.matmul(x,[[2.0]]) + 5.0

#构建线性模型
weight = tf.Variable(tf.random_normal([1,1]),name='w',
                     trainable=True)#是否为训练模式
bias = tf.Variable(0.0,name='b',trainable=True)

pred_y = tf.matmul(x,weight) + bias

#损失函数，均方误差
loss = tf.reduce_mean(tf.square(y-pred_y))
#梯度下降优化器
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化
init_op = tf.global_variables_initializer()

#收集标量
tf.summary.scalar('losses',loss)
#合并摘要
merged = tf.summary.merge_all()

#创建模型保存对象
saver = tf.train.Saver()
#运行
with tf.Session() as sess:
    sess.run(init_op)
    print('weight:{},bias:{}'.format(weight.eval(),
                                     bias.eval()))

    #创建文件写入对象
    fw = tf.summary.FileWriter('../summary/',graph=sess.graph)
    #在训练之前，检查是否有模型保存，如果有则加载
    if os.path.exists('../model/linearmodel/checkpoint'):
        saver.restore(sess,'../model/linearmodel/')
    #开始训练
    for i in range(100):
        sess.run(train_op)
        #执行合并
        summary = sess.run(merged)
        fw.add_summary(summary,i+1)
        print('epoch:{},weight:{},bias:{}'.format(i+1,
                                                  weight.eval(),
                                                  bias.eval()))

    #保存模型
    saver.save(sess,'../model/linearmodel/')



