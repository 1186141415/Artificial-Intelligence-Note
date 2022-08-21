'''
简单的线性回归
'''
import paddle.fluid as fluid
import numpy as np

x_data = np.array([[0.5], [0.6], [0.8], [1.1], [1.4]])
y_data = np.array([[5.0], [5.5], [6.0], [6.8], [7.1]])

# 定义数据占位符
x = fluid.layers.data(name='xxx', shape=[1], dtype='float64')
y = fluid.layers.data(name='yyy', shape=[1], dtype='float64')
# 搭建神经网络
pred_y = fluid.layers.fc(input=x,  # 输入数据
                         size=1,  # 神经元数量
                         act=None)  # 激活函数

# 损失函数:均方误差
cost = fluid.layers.square_error_cost(input=pred_y,  # 预测值
                                      label=y)  # 真实值
avg_cost = fluid.layers.mean(cost)
# 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

# 运行
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化

for i in range(500):
    outs = exe.run(program=fluid.default_main_program(),
                   feed={'xxx': x_data, 'yyy': y_data},
                   fetch_list=[pred_y, avg_cost])
    print('轮数:{},损失值:{}'.format(i+1,outs[1][0]))
