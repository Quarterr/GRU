import torch
from torch.utils.data import TensorDataset
import numpy as np
from torch.utils.data import TensorDataset
import scipy.io as sio


class Config:
    time_step = 9  # 时间步长，就是利用多少时间窗口
    batch_size = 300  # 批次大小
    feature_size = 4  # 每个步长对应的特征数量
    output_size = 1  # 输出6维向量。3个位置和对应的导数
    epochs = 20  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.003  # 初始化学习率
    model_name = 'tst'  # 模型名称
    save_path = './{}.pth'.format(model_name)  # 最优模型保存路径


config = Config()


# 构建时序数据
def preprocess_data(flag = 0):
    config = Config()

    # 1,处理训练数据 标准化
    # datafile = sio.loadmat('C:/Users/HP/Desktop/CEG5003/dataset/Dataset_nurbs_1/training_set_transformer.mat')
    datafile = sio.loadmat('new_data.mat')

    train_set = None
    for i in range(1, 100):
        data = datafile.get(f'series_training{i}')
        if train_set is None:
            train_set = data
        else:
            train_set = np.vstack((train_set, data))

    # 数据标准化 对每个特征分别标准化
    # 获取数据集的形状
    num_samples, num_features = train_set.shape

    # 初始化一个与数据集形状相同的数组用于存储标准化后的数据
    normalized_train_set = np.zeros_like(train_set)

    # 对每个特征分别进行标准化
    for feature_index in range(num_features):
        feature_data = train_set[:, feature_index]
        mean = np.mean(feature_data)
        std = np.std(feature_data)
        normalized_feature = (feature_data - mean) / std
        normalized_train_set[:, feature_index] = normalized_feature

    # 构建时序数据   2000个一组构建（series）
    train_dataX = []
    train_dataY = []
    group_size = 2000  # 每组包含2000个样本

    for start_idx in range(0, len(normalized_train_set), group_size):
        end_idx = start_idx + group_size
        group_data = normalized_train_set[start_idx:end_idx]

        for index in range(len(group_data) - config.time_step):
            train_dataX.append(group_data[index: index + config.time_step, 0: config.feature_size])
            if (flag == 0):
                train_dataY.append(group_data[index + config.time_step, config.feature_size: config.feature_size + config.output_size])
            else:
                train_dataY.append(group_data[index: index + config.time_step, config.feature_size: config.feature_size + config.output_size])

    train_dataX = np.array(train_dataX)
    train_dataY = np.array(train_dataY)

    # 4, 构建data_loader
    x_train = train_dataX.reshape(-1, config.time_step, config.feature_size)
    if (flag == 0):
        y_train = train_dataY.reshape(-1, 1, config.output_size)
    else:
        y_train = train_dataY.reshape(-1, config.time_step, config.output_size)

    x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).to(torch.float32)

    train_data = TensorDataset(x_train_tensor, y_train_tensor)

    # 处理测试集 不需要标准化
    # datafile2 = sio.loadmat('C:/Users/HP/Desktop/CEG5003/dataset/Dataset_nurbs_1/training_set_transformer.mat')
    datafile2 = sio.loadmat('new_data.mat')

    test_dataX = []
    test_dataY = []
    for i in range(150, 186):
        data = datafile2.get(f'series_training{i}')
        # 构建时序数据
        for index in range(len(data) - config.time_step):
            test_dataX.append(data[index: index + config.time_step, 0: config.feature_size])
            test_dataY.append(data[index + config.time_step][config.feature_size: config.feature_size + config.output_size])
    test_dataX = np.array(test_dataX)
    test_dataY = np.array(test_dataY)

    x_test = test_dataX.reshape(-1, config.time_step, config.feature_size)
    y_test = test_dataY.reshape(-1, 1, config.output_size)

    x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    # 将数据加载成迭代器
    train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, config.batch_size, shuffle=False)

    return train_loader, test_loader
