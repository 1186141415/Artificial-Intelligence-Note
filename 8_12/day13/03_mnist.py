'''
手写体识别
模型：全连接神经网络
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import pylab
import numpy as np

# 数据准备
mnist = input_data.read_data_sets('../MNIST_data/',
                                  one_hot=True)  # 独热编码
# 定义占位符，表示图像数据，类别标签
x = tf.placeholder('float32', shape=[None, 784])
y = tf.placeholder('float32', shape=[None, 10])

# 构建模型
weight = tf.Variable(tf.random_normal([784, 10]))
bias = tf.Variable(tf.zeros([10]))
# 预测值
pred_y = tf.nn.softmax(tf.matmul(x, weight) + bias)
# 损失函数:交叉熵
cross_entropy = -tf.reduce_sum(y * tf.log(pred_y), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)
# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 模型保存对象
saver = tf.train.Saver()
model_save_path = '../model/mnist/'
batch_size = 100  # 批次大小

# 执行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化

    if os.path.exists('../model/mnist/checkpoint'):
        saver.restore(sess, model_save_path)
    # 开始训练
    for epoch in range(10):
        total_batch = int(mnist.train.num_examples / batch_size)
        total_cost = 0.0  # 每一轮的总损失值
        for i in range(total_batch):
            # 拿到一次批次的样本数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            params = {x: batch_xs, y: batch_ys}
            o, cost_res = sess.run([optimizer, cost],
                                   feed_dict=params)
            total_cost += cost_res  # 600个批次的损失值
        avg_cost = total_cost / total_batch  # 一轮的平均损失值
        print('epoch:{},cost:{}'.format(epoch, avg_cost))
    print('训练完成...')
    # 模型评估
    corr_pred = tf.equal(tf.arg_max(y, 1),  # 真实类别
                         tf.arg_max(pred_y, 1))  # 预测类别
    # 精度
    acc = tf.reduce_mean(tf.cast(corr_pred, 'float32'))

    print('精度:', acc.eval({x: mnist.test.images,
                           y: mnist.test.labels}))
    # 保存模型
    save_path = saver.save(sess, model_save_path)
    print('模型保存成功:', save_path)

# 从测试机中抽取两张图像，执行预测
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化
    saver.restore(sess, model_save_path)  # 加载模型
    # 从测试集读到2张图像
    test_images, test_labels = mnist.test.next_batch(2)
    output = tf.arg_max(pred_y, 1)  # 预测类别
    out_val, predv = sess.run([output, pred_y],
                              feed_dict={x: test_images})
    print('预测类别为:',out_val)
    print('真实类别为:',np.argmax(test_labels,axis=1))
    print('预测概率为:',np.max(predv,axis=1))

    #显示图像
    img = test_images[0]
    img = img.reshape(-1,28)
    pylab.imshow(img)
    pylab.show()

    img = test_images[1]
    img = img.reshape(-1, 28)
    pylab.imshow(img)
    pylab.show()
