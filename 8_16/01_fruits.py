############################# 预处理部分 ################################
import os

name_dict = {"apple": 0,
             "banana": 1,
             "grape": 2,
             "orange": 3,
             "pear": 4}
data_root_path = "data/fruits/"  # 数据样本所在目录
test_file_path = data_root_path + "test.txt"  # 测试文件路径
train_file_path = data_root_path + "train.txt"  # 训练文件路径
name_data_list = {}  # 记录每个类别有哪些图片  key:水果名称  value:图片路径构成的列表


# 将图片路径存入name_data_list字典中
def save_train_test_file(path, name):
    if name not in name_data_list:  # 该类别水果不在字典中，则新建一个列表插入字典
        img_list = []
        img_list.append(path)  # 将图片路径存入列表
        name_data_list[name] = img_list  # 将图片列表插入字典
    else:  # 该类别水果在字典中，直接添加到列表
        name_data_list[name].append(path)


# 遍历数据集下面每个子目录，将图片路径写入上面的字典
dirs = os.listdir(data_root_path)  # 列出数据集目下所有的文件和子目录
for d in dirs:
    full_path = data_root_path + d  # 拼完整路径

    if os.path.isdir(full_path):  # 是一个子目录
        imgs = os.listdir(full_path)  # 列出子目录中所有的文件
        for img in imgs:
            save_train_test_file(full_path + "/" + img,  # 拼图片完整路径
                                 d)  # 以子目录名称作为类别名称
    else:  # 文件
        pass

# 将name_data_list字典中的内容写入文件
## 清空训练集和测试集文件
with open(test_file_path, "w") as f:
    pass

with open(train_file_path, "w") as f:
    pass

# 遍历字典，将字典中的内容写入训练集和测试集
for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)  # 获取每个类别图片数量
    print("%s: %d张" % (name, num))
    # 写训练集和测试集
    for img in img_list:
        if i % 10 == 0:  # 每10笔写一笔测试集
            with open(test_file_path, "a") as f:  # 以追加模式打开测试集文件
                line = "%s\t%d\n" % (img, name_dict[name])  # 拼一行
                f.write(line)  # 写入文件
        else:  # 训练集
            with open(train_file_path, "a") as f:  # 以追加模式打开测试集文件
                line = "%s\t%d\n" % (img, name_dict[name])  # 拼一行
                f.write(line)  # 写入文件

        i += 1  # 计数器加1

print("数据预处理完成.")

#################### 模型定义、训练 ###########################
import paddle
import paddle.fluid as fluid
import numpy
import sys
import os
from multiprocessing import cpu_count
import time
import matplotlib.pyplot as plt


def train_mapper(sample):
    """
    传入一行文本样本，解析图片路径，读取图像数据
    :param sample: 元组， 格式 ： (图片路径, 类别)
    :return: 返回经过归一化处理的图像数据、类别
    """
    img, label = sample  # img路径，label为类别
    # 读取图像
    img = paddle.dataset.image.load_image(img)
    # 对图像进行缩放、裁剪
    img = paddle.dataset.image.simple_transform(
        im=img,  # 输入图像
        resize_size=128,  # 缩放大小
        crop_size=128,  # 裁剪大小
        is_color=True,  # 是否为彩色图像
        is_train=True)  # 训练模式(训练模式下会做随机裁剪)
    # 图像数据归一化(将像素值转换0~1)
    # 使用梯度下降法优化的模型必须要进行归一化处理
    img = img.astype("float32") / 255.0
    return img, label


# 读取训练集reader
def train_r(trans_list, buffered_size=1024):
    def reader():
        with open(trans_list, "r") as f:  # 打开文件
            lines = [line.strip() for line in f]
            for line in lines:
                line = line.replace("\n", "")  # 去换行符
                img_path, lbl = line.split("\t")  # 拆分
                yield img_path, int(lbl)

    return paddle.reader.xmap_readers(
        train_mapper,  # reader读取的数据进行下一步处理
        reader,  # 读取器
        cpu_count(),  # 线程数量，和CPU数量一致
        buffered_size)  # 缓冲区大小(预分配内存数量)


def create_CNN(image, type_size):
    """
    定义CNN
    :param image: 图像数据(张量)
    :param type_size: 类别数量
    :return: 模型计算结果(模型预测值)
    """
    # 第一组 卷积/激活/池化/dropout
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=image,  # 输入
        filter_size=3,  # 卷积核大小3*3
        num_filters=32,  # 卷积核数量
        pool_size=2,  # 池化区域大小 2*2
        pool_stride=2,  # 池化步长
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_1,
                                dropout_prob=0.5)  # 丢弃率

    # 第二组 卷积/激活/池化/dropout
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 输入
        filter_size=3,  # 卷积核大小3*3
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 池化区域大小 2*2
        pool_stride=2,  # 池化步长
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_2,
                                dropout_prob=0.5)  # 丢弃率

    # 第三组 卷积/激活/池化/dropout
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 输入
        filter_size=3,  # 卷积核大小3*3
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 池化区域大小 2*2
        pool_stride=2,  # 池化步长
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_3,
                                dropout_prob=0.5)  # 丢弃率

    # fc1
    fc = fluid.layers.fc(input=drop,
                         size=512,
                         act="relu")
    drop = fluid.layers.dropout(x=fc,
                                dropout_prob=0.5)
    # 输出层
    predict = fluid.layers.fc(input=drop,
                              size=type_size,  # 类别数量
                              act="softmax")  # 输出层采用softmax
    return predict


BATCH_SIZE = 32  # 批次大小

train_reader = train_r(train_file_path)
random_train_reader = paddle.reader.shuffle(
    reader=train_reader,
    buf_size=1300)  # 随机reader
batch_train_reader = paddle.batch(random_train_reader,
                                  batch_size=BATCH_SIZE)
# 输入、输出
image = fluid.layers.data(name="image",  # 张量名称
                          shape=[3, 128, 128],  # 形状
                          dtype="float32")  # 张量元素类型
label = fluid.layers.data(name="label",
                          shape=[1],
                          dtype="int64")

predict = create_CNN(image=image, type_size=5)

# 损失函数
cost = fluid.layers.cross_entropy(input=predict,  # 预测值
                                  label=label)  # 真实值
avg_cost = fluid.layers.mean(cost)  # 目标函数
# 优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)  # 指定优化的目标函数
# 准确率
accuracy = fluid.layers.accuracy(input=predict,  # 预测值
                                 label=label)  # 真实值

# 执行器
place = fluid.CUDAPlace(0)  # GPU
exe = fluid.Executor(place)  # 执行器
exe.run(fluid.default_startup_program())  # 初始化
# feeder
feeder = fluid.DataFeeder(feed_list=[image, label],  # 喂入的数据
                          place=place)

costs = []  # 记录损失值
accs = []  # 记录准确率
times = 0  # 计数器
batches = []  # 记录迭代次数

# 开始训练
for epoch in range(10):  # 外层控制轮次
    # enumerate：对括号中的可迭代对象元素自动编号
    # 内层循环控制批次
    for bat_id, data in enumerate(batch_train_reader()):
        times += 1

        train_cost, train_acc = exe.run(
            fluid.default_main_program(),  # 执行的program
            feed=feeder.feed(data),  # 需要喂入的数据
            fetch_list=[avg_cost, accuracy])  # 指定返回哪些操作结果

        if bat_id % 20 == 0:  # 每20个批次打印一笔
            print("epoch:%d, batch:%d, cost:%f, acc:%f" %
                  (epoch, bat_id, train_cost[0], train_acc[0]))
            accs.append(train_acc[0])  # 记录准确率
            costs.append(train_cost[0])  # 记录损失值
            batches.append(times)  # 记录迭代批次数量
print("训练结束.")

# 保存模型
model_save_dir = "model/"  # 模型保存目录
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)  # 目录不存在，则创建
# 保存推理模型(推理模型只能用于预测，经过裁剪、优化后的模型)
# 用与增量训练的模型是完整的(save_persistable函数)
fluid.io.save_inference_model(
    dirname=model_save_dir, # 模型保存路径
    feeded_var_names=["image"], # 模型推理张量名称
    target_vars=[predict], # 模型预测结果
    executor=exe) # 执行器
print("保存推理模型结束.")

# 训练过程可视化
plt.title("training", fontsize=24)
plt.xlabel("iter", fontsize=20)
plt.ylabel("cost/acc", fontsize=20)
plt.plot(batches, costs, color='red', label="Training Cost")
plt.plot(batches, accs, color='green', label="Training Acc")
plt.legend()
plt.grid()
plt.savefig("train.png")
plt.show()

