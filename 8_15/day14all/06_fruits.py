'''
利用CNN卷积神经网络，实现水果分类
'''

#####################数据预处理############################
import os

name_dict = {'apple': 0, 'banana': 1, 'grape': 2, 'orange': 3, 'pear': 4}
data_root_path = './data/fruits/'
train_file_path = data_root_path + 'train.txt'
test_file_path = data_root_path + 'test.txt'
name_data_list = {}  # 存放每个类别的图片路径
# {'apple':['0.jpg','1.jpg',.....],'banana':[.....]}


# 遍历每个子目录，拼接完整路径,并加入字典中
dirs = os.listdir(data_root_path)


def save_train_test_file(path, name):
    if name not in name_data_list:
        img_list = []
        img_list.append(path)
        name_data_list[name] = img_list
    else:
        name_data_list[name].append(path)


for d in dirs:
    full_path = data_root_path + d  # ../fruits/apple

    if os.path.isdir(full_path):
        imgs = os.listdir(full_path)  # 子文件夹中所有的图片名称
        for img in imgs:
            all_path = full_path + '/' + img
            save_train_test_file(all_path,  # 完整路径
                                 d)  # 以文件夹名作为键
    else:
        pass

# 生成训练集和测试集文件
with open(train_file_path, 'w') as f:
    pass
with open(test_file_path, 'w') as f:
    pass

# 遍历字典，划分训练集和测试集  #格式:    路径\t类别
for name, img_list in name_data_list.items():
    print('{}类别:{}张！'.format(name, len(img_list)))
    i = 0
    for img in img_list:
        if i % 10 == 0:
            #写入测试集
            with open(test_file_path,'a') as f:
                #拼接一行   路径\t类别
                line = '%s\t%d\n' % (img,name_dict[name])
                f.write(line)
        else:
            #写入训练集
            with open(train_file_path,'a') as f:
                # 拼接一行   路径\t类别
                line = '%s\t%d\n' % (img, name_dict[name])
                f.write(line)
        i += 1

print('数据预处理完成')
