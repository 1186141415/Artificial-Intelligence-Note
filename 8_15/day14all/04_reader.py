'''
读取器
'''
import paddle


# 原始读取器
def reader_creator(file_path):
    def reader():
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                yield line.replace('\n', '')

    return reader


reader = reader_creator('test.txt')
# 随机读取器
shuffle_reader = paddle.reader.shuffle(reader,
                                       1024)  # 缓冲区大小
# 批量随机读取器
batch_reader = paddle.batch(shuffle_reader,
                            3)  # 批次大小

# for i in reader():
# for i in shuffle_reader():
for i in batch_reader():
    print(i)
