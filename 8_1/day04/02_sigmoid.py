'''
将预测值带入到sigmoid中
'''
# 将-10,10 拆成200个，作为预测值
import numpy as np
import matplotlib.pyplot as plt

ys = np.linspace(-10,10,200)
res = 1 / (1 + np.e**-ys)

plt.plot(ys,res,color='orangered')
plt.show()


