'''
服饰识别
模型：卷积神经网络
'''
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


class FashionMnist:
    out_feature1 = 12  # 第一组卷积核数量
    out_feature2 = 24  # 第二组卷积核数量
    con_neurons = 512  # 第一个全连接神经元数量

    def __init__(self, path):
        self.data = read_data_sets(path, one_hot=True)
        self.sess = tf.Session()

    def init_weight_var(self, shape):
        '''根据指定形状，初始化权重'''
        # 截尾正态分布
        init_val = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_val)

    def init_bias_var(self, shape):
        '''根据指定形状，初始化偏置'''
        init_val = tf.constant(1.0, shape=shape)
        return tf.Variable(init_val)

    def conv2d(self, x, w):
        '''
        二维卷积
        :param x:输入数据
        :param w: 卷积核
        :return: 卷积结果
        '''
        return tf.nn.conv2d(x,  # 输入
                            w,  # 卷积核
                            strides=[1, 1, 1, 1],  # 卷积步长
                            padding='SAME')  # 同维卷积

    def max_pool_2x2(self, x):
        '''
        2x2区域最大池化
        :param x: 输入数据(卷积激活之后的结果)
        :return:池化结果
        '''
        return tf.nn.max_pool(x,  # 输入数据
                              ksize=[1, 2, 2, 1],  # 池化区域
                              strides=[1, 2, 2, 1],  # 池化步长
                              padding='SAME')

    def create_conv_pool_layer(self, input, input_feature, out_feature):
        '''
        卷积池化组
        :param input: 输入数据
        :param input_feature:输入特征数量
        :param out_feature: 输出特征数量
        :return: 计算结果
        '''
        filter_w = self.init_weight_var([5, 5, input_feature, out_feature])
        b_conv = self.init_bias_var([out_feature])
        h_conv = tf.nn.relu(self.conv2d(input, filter_w) + b_conv)
        h_pool = self.max_pool_2x2(h_conv)
        return h_pool

    def create_fc_layer(self, h_pool_flat, input_feature, con_neurons):
        '''
        第一个全连接层
        :param h_pool_flat:经过拉伸1维之后的输入数据
        :param input_feature:输入特征数量
        :param con_neurons:神经元数量
        :return:计算结果
        '''
        w_fc = self.init_weight_var([input_feature, con_neurons])
        b_fc = self.init_bias_var([con_neurons])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, w_fc) + b_fc)
        return h_fc1

    def build(self):
        '''
        组建CNN
        :return:
        '''
        # 定义输入和输出的占位符
        self.x = tf.placeholder('float32', shape=[None, 784])
        self.y = tf.placeholder('float32', shape=[None, 10])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        # 第一组卷积池化
        h_pool1 = self.create_conv_pool_layer(x_image, 1, self.out_feature1)
        # 第二组卷积池化
        h_pool2 = self.create_conv_pool_layer(h_pool1, self.out_feature1, self.out_feature2)
        # 全连接
        # 输入特征数量
        h_pool2_flat_features = 7 * 7 * self.out_feature2
        # 输出数据，二维
        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_flat_features])
        h_fc1 = self.create_fc_layer(h_pool2_flat, h_pool2_flat_features, self.con_neurons)
        # 丢弃层
        self.keep = tf.placeholder('float32')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=self.keep)
        # 输出层
        w_fc = self.init_weight_var([self.con_neurons, 10])
        b_fc = self.init_bias_var([10])
        pred_y = tf.matmul(h_fc1_drop, w_fc) + b_fc
        # 损失函数,交叉熵
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y,  # 真实值
                                                       logits=pred_y)  # 预测值
        cross_entropy = tf.reduce_mean(loss)
        # 准确率
        corr_pred = tf.equal(tf.argmax(self.y, 1),
                             tf.argmax(pred_y, 1))

        self.acc = tf.reduce_mean(tf.cast(corr_pred, 'float32'))
        # 优化器(自适应梯度下降优化器)
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.minimize(cross_entropy)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        batch_size = 100
        print('开始训练.....')
        for i in range(10):
            total_batch = int(self.data.train.num_examples / batch_size)
            for j in range(total_batch):
                train_data = self.data.train.next_batch(batch_size)
                params = {self.x: train_data[0],
                          self.y: train_data[1],
                          self.keep: 0.5}

                t, accuracy = self.sess.run([self.train_op, self.acc],
                                            feed_dict=params)
                if j % 100 == 0:
                    print('轮数:{},批次:{},精度:{}'.format(i,j,accuracy))

    def metrics(self,x,y,keep):
        params = {self.x:x,self.y:y,self.keep:keep}
        test_acc = self.sess.run(self.acc,feed_dict=params)
        print('测试集精度:',test_acc)

    def close(self):
        self.sess.close()

if __name__ == '__main__':
    fmnist = FashionMnist('../fashionmnist/')
    fmnist.build()#搭建网络
    fmnist.train()#执行训练
    test_x,test_y = fmnist.data.test.next_batch(100)
    fmnist.metrics(test_x,test_y,0.5)
    fmnist.close()



