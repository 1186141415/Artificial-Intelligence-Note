'''
读取图像文件示例
'''
import tensorflow as tf
import os


def read_img(filelist):
    # 构建文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 构建文件读取器
    reader = tf.WholeFileReader()
    # 读取数据
    name, val = reader.read(file_queue)
    # 解码
    img = tf.image.decode_jpeg(val)
    # 批处理
    img_resized = tf.image.resize(img, [200, 200])
    img_resized.set_shape([200, 200, 3])
    img_bat = tf.train.batch([img_resized],
                             batch_size=10,
                             num_threads=1)
    return img_bat

if __name__ == '__main__':
    #构建文件列表
    dir_name = './test_img/'
    file_names = os.listdir(dir_name)
    file_list = []
    for f in file_names:
        file_list.append(os.path.join(dir_name,f))

    imgs = read_img(file_list)
    #执行
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord=coord)
        res = sess.run(imgs)
        #请求停止，回收资源
        coord.request_stop()
        coord.join(threads)

import matplotlib.pyplot as plt

fig = plt.figure('Imshow',facecolor='lightgray')

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(res[i].astype('int32'))
    plt.xticks([])
    plt.yticks([])
plt.tight_layout() #紧凑式布局
plt.show()












