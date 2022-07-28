'''
使用Python的代码实现梯度下降
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
y = np.array([5.0, 5.5, 6.0, 6.8, 7.1])

# 设定模型参数初始值
w1 = 1  # 权重的初始值一般为随机数
w0 = 1  # 偏置的初始值一般为0或者1
learning_rate = 0.2  # 学习率
epoch = 300  # 训练轮数


w1s = []
w0s = []
losses = []
epoches = []
for i in range(epoch):
    loss = ((w1 * x + w0 - y) ** 2).sum() / 2
    print('轮数:{:3},w1:{:.8f},w0:{:.8f},loss:{:.8f}'.format(i + 1,
                                                           w1,
                                                           w0,
                                                           loss))

    w1s.append(w1)
    w0s.append(w0)
    losses.append(loss)
    epoches.append(i+1)

    d1 = (x * (w1 * x + w0 - y)).sum()
    d0 = (w0 + w1 * x - y).sum()
    w1 = w1 - learning_rate * d1
    w0 = w0 - learning_rate * d0

# print('w1:{},w0:{}'.format(w1,w0))
pred_y = w1 * x + w0  # 预测值

# 回归线可视化
plt.plot(x, pred_y, color='orangered')
plt.scatter(x, y)

#模型参数及损失函数可视化
plt.figure('training')
plt.subplot(3,1,1)
plt.plot(epoches,w1s,color='dodgerblue')

plt.subplot(3,1,2)
plt.plot(epoches,w0s,color='dodgerblue')

plt.subplot(3,1,3)
plt.plot(epoches,losses,color='orangered')

plt.show()
