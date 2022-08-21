'''
波士顿房屋价格预测
'''
import sklearn.datasets as sd  # 数据集合
import sklearn.model_selection as ms  # 模型选择
import sklearn.metrics as sm #模型评估
import sklearn.linear_model as lm #线性模型
import sklearn.preprocessing as sp #数据预处理
import sklearn.pipeline as pl #数据管线
import sklearn.tree as st #决策树
import sklearn.ensemble as se #集成学习

boston = sd.load_boston()

# print(boston.keys())
# print(type(boston))
# print(boston.filename)
# print(boston.DESCR)
# print(boston.feature_names)
# print(boston.data.shape) #输入
# print(boston.target.shape)#输出

# 整理输入和输出
x = boston.data
y = boston.target

# 划分训练集和测试集
train_x, \
test_x, \
train_y, \
test_y = ms.train_test_split(x, y,
                             test_size=0.1,
                             random_state=7)
# print(data[0].shape) #训练集的x
# print(data[1].shape) #测试集的x
# print(data[2].shape) #训练集的y
# print(data[3].shape) #测试机的y

#根据训练集 训练模型   根据测试机去评估模型

# 构建，线性回归，岭回归，多项式回归。告诉我那个模型好
# 只有构建模型不同，其他的都一样

def get_model(name,model):
    print('----------------',name,'-------------')
    model.fit(train_x,train_y)
    pred_train_y = model.predict(train_x)
    pred_test_y = model.predict(test_x)
    print(name,':训练集得分:',sm.r2_score(train_y,pred_train_y))
    print(name,':测试集得分:',sm.r2_score(test_y,pred_test_y))

model_dict = {'线性回归':lm.LinearRegression(),
              '岭回归':lm.Ridge(alpha=100),
              '多项式回归':pl.make_pipeline(sp.PolynomialFeatures(2),
                                           lm.LinearRegression()),
              '单颗决策树':st.DecisionTreeRegressor(max_depth=6)}
#
# for name,obj in model_dict.items():
#     get_model(name,obj)


# Adaboost回归
model = st.DecisionTreeRegressor(max_depth=6)
model = se.AdaBoostRegressor(model,
                             n_estimators=400)
model.fit(train_x,train_y)
pred_train_y = model.predict(train_x)
pred_test_y = model.predict(test_x)
print('训练集得分:',sm.r2_score(train_y,pred_train_y))
print('测试集得分:',sm.r2_score(test_y,pred_test_y))







