'''文本文件读取示例'''
import tensorflow as tf
import os


def read_csv(filelist):
    # 构建文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 构建文件读取器
    reader = tf.TextLineReader()
    # 读取数据
    name, val = reader.read(file_queue)
    # 解码
    records = [['None'], ['None']]
    example, label = tf.decode_csv(val, record_defaults=records)
    # 批处理
    exam_bat, lab_bat = tf.train.batch([example, label],
                                       batch_size=5,
                                       num_threads=1)
    return exam_bat,lab_bat

if __name__ == '__main__':
    #构建文件列表
    dir_name = './test_data/'
    file_names = os.listdir(dir_name)
    file_list = []
    for f in file_names:
        file_list.append(os.path.join(dir_name,f))

    example,label = read_csv(file_list)

    #开启session，执行
    with tf.Session() as sess:
        #线程协调器
        coord = tf.train.Coordinator()
        #开启读取文件线程
        threads = tf.train.start_queue_runners(sess,coord=coord)
        print(sess.run([example,label]))
        #等待线程停止，回收资源
        coord.request_stop()
        coord.join(threads)
