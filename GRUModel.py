import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from torch.utils.data import TensorDataset


class Config:
    time_step = 9  # 时间步长，就是利用多少时间窗口
    batch_size = 300  # 批次大小
    feature_size = 3  # 每个步长对应的特征数量，3个特征值
    output_size = 6  # 输出6维向量。3个位置和对应的导数
    epochs = 50  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.0003  # 初始化学习率
    model_name = 'gru'  # 模型名称
    save_path = './{}.pth'.format(model_name)  # 最优模型保存路径


config = Config()

# 1,处理训练数据 标准化
datafile = sio.loadmat('C:/Users/HP/Desktop/CEG5003/dataset/Dataset_nurbs_1/training_set_transformer.mat')

train_set = np.array([])
for i in range(1, 11):
    data = datafile.get(f'series_training{i}')
    if train_set.size == 0:
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
        train_dataX.append(group_data[index: index + config.time_step, 0:3])
        train_dataY.append(group_data[index + config.time_step, 3:9])

train_dataX = np.array(train_dataX)
train_dataY = np.array(train_dataY)

# 4, 构建data_loader
x_train = train_dataX.reshape(-1, config.time_step, config.feature_size)
y_train = train_dataY.reshape(-1, 1, 6)

x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)

train_data = TensorDataset(x_train_tensor, y_train_tensor)

# 处理测试集 不需要标准化
datafile2 = sio.loadmat('C:/Users/HP/Desktop/CEG5003/dataset/Dataset_nurbs_1/training_set_transformer.mat')

test_dataX = []
test_dataY = []
for i in range(13, 16):
    data = datafile2.get(f'series_training{i}')
    # 构建时序数据
    for index in range(len(data) - config.time_step):
        test_dataX.append(data[index: index + config.time_step, 0: 3])
        test_dataY.append(data[index + config.time_step][3:9])
test_dataX = np.array(test_dataX)
test_dataY = np.array(test_dataY)

x_test = test_dataX.reshape(-1, config.time_step, config.feature_size)
y_test = test_dataY.reshape(-1, 1, 6)

x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, config.batch_size, shuffle=True)


# 5.定义GRU网络  两层gru unit分别是200和1（6） 单轴论文 --修改第二层为全连接层 200-6
# 两种全连接层，最后一步去映射或者整个结果去映射
class GRU(nn.Module):
    def __init__(self, feature_size):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(feature_size, 32, batch_first=True, num_layers=1)
        self.fc = nn.Linear(32, 6)

    def forward(self, x, hidden=None):
        output, _ = self.gru1(x)
        return self.fc(output[:, -1, :])
        # b, s, h = output.shape
        # fcInput = output.reshape(s * b, h)
        # print(fcInput.shape)
        # fcOutput = self.fc(fcInput)
        # outcome = fcOutput.reshape(b, s, -1)
        # return outcome[:, -1, :]


# # 5.定义GRU网络  两层gru unit分别是200和1（6） 单轴论文
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

# # 5.定义GRU网络  5层gru unit分别是32，64，64，64，1（6） 多轴论文
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


# # 5.定义GRU网络  5层gru unit分别是32，64，64，64，1（6） 多轴论文 -- 修改最后为全连接层64-6
# # 过拟合
# class GRU(nn.Module):
#     def __init__(self, feature_size):
#         super(GRU, self).__init__()
#         self.gru1 = nn.GRU(feature_size, 32, batch_first=True)  ##32 unit
#         self.gru2 = nn.GRU(32, 64, batch_first=True)  ##64 unit
#         self.gru3 = nn.GRU(64, 64, batch_first=True)  ##64 unit
#         self.gru4 = nn.GRU(64, 64, batch_first=True)  ##64 unit
#         self.fc = nn.Linear(64, 6)
#
#     def forward(self, x, hidden=None):
#         out1, _ = self.gru1(x)
#         out2, _ = self.gru2(out1)
#         out3, _ = self.gru3(out2)
#         out4, _ = self.gru4(out3)
#         # 提取最后一个时间步的输出
#         prediction = out4[:, -1, :]
#
#         output = self.fc(prediction)
#         return output


# 6, 定义model, 损失函数，优化器和学习率
model = GRU(config.feature_size)  # 定义GRU网络
loss_function = nn.L1Loss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # 定义优化器
scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

# 7.模型训练 并且保存效果最优模型
prev_loss = 1000
for epoch in range(config.epochs):
    # 动态调整学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    running_loss = 0.0
    for data in train_loader:
        x_train, y_train = data
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 6))
        loss.backward()
        optimizer.step()
        if loss < prev_loss:
            torch.save(config.save_path, 'gru_model.pt')  # save model parameters to files
            prev_loss = loss
        running_loss += loss.item()
    print('epoch:', epoch, 'loss:', running_loss / len(train_loader), "current_lr:", current_lr)

print('Finished Training')


# 8. 模型测试
model.eval()  # 将模型设置为评估模式

# 存储预测结果和真实值
predictions = []
true_values = []

with torch.no_grad():  # 关闭梯度计算
    for data in test_loader:
        x_test_batch, y_test_batch = data
        y_pred_batch = model(x_test_batch)
        predictions.append(y_pred_batch)
        true_values.append(y_test_batch)

# 将预测结果和真实值转换为 NumPy 数组
predictions = torch.cat(predictions, dim=0).numpy()
true_values = torch.cat(true_values, dim=0).numpy().reshape(-1, 6)

# 初始化一个变量来统计预测正确的数量
correct_predictions = 0

# 顺便保存1-norm值的数据
predictions_l1_norm = []
true_values_l1_norm = []
for i in range(len(predictions)):
    prediction_l1_norm = np.linalg.norm(predictions[i])
    true_value_l1_norm = np.linalg.norm(true_values[i])

    if abs(prediction_l1_norm - true_value_l1_norm) < 0.001:
        correct_predictions += 1

    predictions_l1_norm.append(prediction_l1_norm)
    true_values_l1_norm.append(true_value_l1_norm)

# 计算准确率
accuracy = correct_predictions / len(predictions)
print("Test Accuracy:", accuracy)


# 9. 绘制预测结果
plot_size = 100  # 可视化的数据点数量
plt.figure(figsize=(12, 8))
plt.plot(predictions_l1_norm[:plot_size], "b", label="Predicted_norm")
plt.plot(true_values_l1_norm[:plot_size], "r", label="True_norm")
plt.legend()
plt.show()


# 看一个预测值吧
print("测试集输入", x_test_tensor[77])
print("真实值", y_test_tensor[77])
print("预测值", model(x_test_tensor)[77])
