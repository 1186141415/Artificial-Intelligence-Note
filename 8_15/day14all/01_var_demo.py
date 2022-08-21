'''
paddle简单的变量演示
'''
import paddle.fluid as fluid
import numpy as np

# 创建常量类型的变量
x = fluid.layers.fill_constant(shape=[1], dtype='float32', value=100.0)
y = fluid.layers.fill_constant(shape=[1], dtype='float32', value=200.0)
res = x + y

# 执行
place = fluid.CPUPlace()
exe = fluid.Executor(place)
result = exe.run(program=fluid.default_main_program(),
                 fetch_list=[res])

print(result) #列表套数组
print(result[0])#数组
print(result[0][0])