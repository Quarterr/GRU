import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import scipy.io as sio


class Config:
    timestep = 9  # 时间步长，就是利用多少时间窗口
    batch_size = 200  # 批次大小
    feature_size = 3  # 每个步长对应的特征数量，3个特征值
    output_size = 6  # 输出6维向量。3个位置和对应的导数
    epochs = 10  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.003  # 学习率
    model_name = 'gru'  # 模型名称
    save_path = './{}.pth'.format(model_name)  # 最优模型保存路径


config = Config()

# 加载数据
datafile = sio.loadmat('C:/Users/HP/Desktop/CEG5003/dataset/Dataset_nurbs_1/training_set_transformer.mat')


scaler = MinMaxScaler()
dataX = []
dataY = []
for i in range(1, 11):
    data = datafile.get(f'series_training{i}')
    # 标准化
    data = scaler.fit_transform(np.array(data))
    # 构建时序数据
    for index in range(len(data) - config.timestep):
        dataX.append(data[index: index + config.timestep, 0: 3])
        dataY.append(data[index + config.timestep][3:9])


print(len(dataX))

dataX = np.array(dataX)
dataY = np.array(dataY)

# 获取训练集大小
train_size = int(np.round(0.8 * dataX.shape[0]))

# 划分训练集、测试集
x_train = dataX[: train_size, :].reshape(-1, config.timestep, config.feature_size)
y_train = dataY[: train_size].reshape(-1, 1, 6)

x_test = dataX[train_size:, :].reshape(-1, config.timestep, config.feature_size)
y_test = dataY[train_size:].reshape(-1, 1, 6)

# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, config.batch_size, shuffle=True)


# # 7.定义GRU网络  两层gru unit分别是200和1（6） 单轴论文
# class GRU(nn.Module):
#     def __init__(self, feature_size):
#         super(GRU, self).__init__()
#         self.gru1 = nn.GRU(feature_size, 200, batch_first=True)
#         self.gru2 = nn.GRU(200, 6, batch_first=True)
#
#     def forward(self, x, hidden=None):
#         out1, _ = self.gru1(x)
#         out2, _ = self.gru2(out1)
#         # 提取最后一个时间步的输出
#         prediction = out2[:, -1, :]
#         return prediction

# # 7.定义GRU网络  两层gru unit分别是200和1（6） 单轴论文 --修改第二层为全连接层 200-6
# class GRU(nn.Module):
#     def __init__(self, feature_size):
#         super(GRU, self).__init__()
#         self.gru1 = nn.GRU(feature_size, 200, batch_first=True)
#         self.fc = nn.Linear(200, 6)
#
#     def forward(self, x, hidden=None):
#         out1, _ = self.gru1(x)
#         # 提取最后一个时间步的输出
#         prediction = out1[:, -1, :]
#         output = self.fc(prediction)
#         return output

# # 7.定义GRU网络  5层gru unit分别是32，64，64，64，1（6） 多轴论文
# class GRU(nn.Module):
#     def __init__(self, feature_size):
#         super(GRU, self).__init__()
#         self.gru1 = nn.GRU(feature_size, 32, batch_first=True) ##32 unit
#         self.gru2 = nn.GRU(32, 64, batch_first=True) ##64 unit
#         self.gru3 = nn.GRU(64, 64, batch_first=True) ##64 unit
#         self.gru4 = nn.GRU(64, 64, batch_first=True) ##64 unit
#         self.gru5 = nn.GRU(64, 6, batch_first=True) ##32 unit
#
#
#     def forward(self, x, hidden=None):
#         out1, _ = self.gru1(x)
#         out2, _ = self.gru2(out1)
#         out3, _ = self.gru3(out2)
#         out4, _ = self.gru4(out3)
#         out5, _ = self.gru5(out4)
#         # 提取最后一个时间步的输出
#         prediction = out5[:, -1, :]
#         return prediction


# 7.定义GRU网络  5层gru unit分别是32，64，64，64，1（6） 多轴论文 -- 修改最后为全连接层64-6
class GRU(nn.Module):
    def __init__(self, feature_size):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(feature_size, 32, batch_first=True)  ##32 unit
        self.gru2 = nn.GRU(32, 64, batch_first=True)  ##64 unit
        self.gru3 = nn.GRU(64, 64, batch_first=True)  ##64 unit
        self.gru4 = nn.GRU(64, 64, batch_first=True)  ##64 unit
        self.fc = nn.Linear(64, 6)

    def forward(self, x, hidden=None):
        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1)
        out3, _ = self.gru3(out2)
        out4, _ = self.gru4(out3)
        # 提取最后一个时间步的输出
        prediction = out4[:, -1, :]

        output = self.fc(prediction)
        return output


model = GRU(config.feature_size)  # 定义GRU网络
loss_function = nn.L1Loss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # 定义优化器

# 8.模型训练 并且保存效果最优模型
prev_loss = 1000
for epoch in range(config.epochs):
    for data in train_loader:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 6))
        loss.backward()
        optimizer.step()

        if loss < prev_loss:
            torch.save(config.save_path, 'gru_model.pt')  # save model parameters to files
            prev_loss = loss
    print('loss:', loss.item())

print('Finished Training')

##----------------- plot -------------------
# plot_size = 200
# plt.figure(figsize=(12, 8))
# plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()[: plot_size]).reshape(-1, 1)), "b")
# plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
# plt.legend()
# plt.show()

print(x_train_tensor[0])
print("预测值")
print(model(x_train_tensor)[0])
print("实际值")
print(y_train_tensor[0])
