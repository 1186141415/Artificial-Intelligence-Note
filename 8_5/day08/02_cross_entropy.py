'''
计算多个类别的交叉熵
'''
import math

y_true = [0,   1,  0,  0,  0]
pred_y = [0.1,0.6,0.1,0.1,0.1]
pred_y2= [0.1,0.7,0.1,0.05,0.05]
pred_y3= [0.1,0.8,0.03,0.03,0.04]

total_entropy = 0.0
total_entropy2 = 0.0
total_entropy3 = 0.0
for i in range(len(y_true)):
    total_entropy += y_true[i] * math.log(pred_y[i])
    total_entropy2 += y_true[i] * math.log(pred_y2[i])
    total_entropy3 += y_true[i] * math.log(pred_y3[i])

print(-total_entropy)
print(-total_entropy2)
print(-total_entropy3)




