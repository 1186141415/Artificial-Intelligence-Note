'''
使用全连接模型，预测波士顿房屋价格
'''
import paddle.fluid as fluid
import paddle
import os
import numpy as np

# 数据准备
reader = paddle.dataset.uci_housing.train()
shuffle_reader = paddle.reader.shuffle(reader, 1024)
batch_reader = paddle.batch(shuffle_reader, 20)

# 搭建模型
x = fluid.layers.data(name='xxx', shape=[13], dtype='float32')
y = fluid.layers.data(name='yyy', shape=[1], dtype='float32')

pred_y = fluid.layers.fc(input=x,
                         size=1,
                         act=None)
# 损失函数
cost = fluid.layers.square_error_cost(input=pred_y,
                                      label=y)
avg_cost = fluid.layers.mean(cost)
# 梯度下降优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.01)
optimizer.minimize(avg_cost)

# 执行
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化

# [([],[]),([],[]),([],[]).....]
# 参数喂入器
feeder = fluid.DataFeeder(feed_list=[x, y], place=place)
for pass_id in range(200):
    i = 0  # 批次大小
    for data in batch_reader():
        i += 1
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])
        if i % 20 == 0:
            print('pass_id:{},cost:{}'.format(pass_id, train_cost[0][0]))

# 保存模型
model_save_path = './model/uci_housing'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

fluid.io.save_inference_model(model_save_path,  # 模型保存路径
                              ['xxx'],  # 需要喂入的参数
                              [pred_y],  # 预测结果从哪取
                              exe)  # 模型参数

# 加载模型
infer_exe = fluid.Executor(place)  # 加载模型的执行器
# 加载到哪个program,需要喂入的参数列表，预测结果列表
infer_program, \
feed_names, \
fetch_names = fluid.io.load_inference_model(model_save_path,
                                            infer_exe)
# 测试机数据进行测试
infer_reader = paddle.batch(paddle.dataset.uci_housing.test(), 200)

test_data = next(infer_reader())

test_x = np.array([data[0] for data in test_data]).astype('float32')
test_y = np.array([data[1] for data in test_data]).astype('float32')

result = infer_exe.run(program=infer_program,
                       feed={feed_names[0]: test_x},
                       fetch_list=fetch_names)

print(result[0][0])
