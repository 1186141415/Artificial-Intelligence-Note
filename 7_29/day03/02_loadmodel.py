'''
封装预测逻辑，构建薪资预测类
'''
import pickle
import numpy as np

class SalaryPredictModel:

    def __init__(self):
        with open('linearRegression.pickle','rb') as f:
            self.model = pickle.load(f)

    def predict_myself(self,exps):
        exps = np.array(exps).reshape(-1,1) #-1为自适应
        return self.model.predict(exps)

model = SalaryPredictModel()
res = model.predict_myself([1.1,2.2,3.3])
print(res)

