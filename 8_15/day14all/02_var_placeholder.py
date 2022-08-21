'''
占位符类型的变量
'''
import paddle.fluid as fluid
import numpy as np

# 占位符变量
x = fluid.layers.data(name='xxx', shape=[2, 3], dtype='float32')
y = fluid.layers.data(name='yyy', shape=[2, 3], dtype='float32')

add = fluid.layers.elementwise_add(x, y)
mul = fluid.layers.elementwise_mul(x, y)  # 对应位置对应相乘

# 执行
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化

data_x = np.arange(1, 7).reshape(2, 3)
data_y = np.arange(1, 7).reshape(2, 3)

out = exe.run(program=fluid.default_main_program(),
              feed={'xxx': data_x, 'yyy': data_y},
              fetch_list=[add, mul])

print(out[0])
print(out[1])


