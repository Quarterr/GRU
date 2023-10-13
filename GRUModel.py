import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import scipy.io as sio
import dataPreprocess

class Config:
    time_step = 9  # 时间步长，就是利用多少时间窗口
    batch_size = 300  # 批次大小
    feature_size = 12  # 每个步长对应的特征数量
    output_size = 3  # 输出6维向量。3个位置和对应的导数
    epochs = 200  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.003  # 初始化学习率
    model_name = 'gru'  # 模型名称
    save_path = './{}.pth'.format(model_name)  # 最优模型保存路径

config = Config()


# 5.定义GRU网络  两层gru unit分别是200和1（6） 单轴论文 --修改第二层为全连接层 200-6
# 两种全连接层，最后一步去映射或者整个结果去映射
class GRU(nn.Module):
    def __init__(self, feature_size):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(feature_size, 1024, batch_first=True, num_layers=4)
        self.gru2 = nn.GRU(1024, 256, batch_first=True, num_layers=1)
        self.fc = nn.Linear(256, 3)

    def forward(self, x, hidden=None):
        output1,_ = self.gru1(x)
        output, _ = self.gru2(output1)
        return self.fc(output[:, -1, :])
        # b, s, h = output.shape
        # fcInput = output.reshape(s * b, h)
        # fcOutput = self.fc(fcInput)
        # outcome = fcOutput.reshape(b, s, -1)
        # return outcome[:, -1, :]
#
# # 5.定义GRU网络  5层gru unit分别是32，64，64，64，1（6） 多轴论文 -- 修改最后为全连接层64-6
# # 过拟合
# class GRU(nn.Module):
#     def __init__(self, feature_size):
#         super(GRU, self).__init__()
#         self.gru1 = nn.GRU(feature_size, 128, batch_first=True)  ##32 unit
#         self.gru2 = nn.GRU(128, 256, batch_first=True)  ##64 unit
#         self.gru3 = nn.GRU(256, 512, batch_first=True)  ##64 unit
#         self.gru4 = nn.GRU(512, 64, batch_first=True)  ##64 unit
#         self.fc = nn.Linear(64, 1)
#
#     def forward(self, x, hidden=None):
#         out1, _ = self.gru1(x)
#         out2, _ = self.gru2(out1)
#         out3, _ = self.gru3(out2)
#         out4, _ = self.gru4(out3)
#         output = self.fc(out4[:, -1, :])
#         return output

if torch.cuda.is_available():
    print("++++++++++++++++++++++++++++++++")
else:
    print("-----------------------------")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = dataPreprocess.preprocess_data()

#6, 定义model, 损失函数，优化器和学习率
model = GRU(config.feature_size).to(device)  # 定义GRU网络
loss_function = nn.L1Loss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # 定义优化器
scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

# 7.模型训练 并且保存效果最优模型
prev_loss = 1000

for epoch in range(config.epochs):
    current_lr = optimizer.param_groups[0]['lr']
    running_loss = 0.0
    for data in train_loader:
        x_train, y_train = data
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 3))
        loss.backward()
        optimizer.step()
        if loss < prev_loss:
            torch.save(model, 'gru_model.pt')
            prev_loss = loss
        running_loss += loss.item()

    # 动态调整学习率
    scheduler.step()
    print('epoch:', epoch, 'loss:', running_loss / len(train_loader), "current_lr:", current_lr)
    # 将损失信息追加到.txt文件中
    with open('loss_results.txt', 'a') as file:
        file.write(f"Epoch {epoch}: Loss {running_loss / len(train_loader)} current_lr: {current_lr}\n")


print('Finished Training')

# 8. 模型测试
model.eval()

# 存储预测结果和真实值
predictions = []
true_values = []

with torch.no_grad():  # 关闭梯度计算
    for data in test_loader:
        x_test_batch, y_test_batch = data
        x_test_batch = x_test_batch.to(device)
        y_test_batch = y_test_batch.to(device)
        y_pred_batch = model(x_test_batch)
        predictions.append(y_pred_batch)
        true_values.append(y_test_batch)

# 将预测结果和真实值转换为 NumPy 数组
predictions = torch.cat(predictions, dim=0).numpy()
true_values = torch.cat(true_values, dim=0).numpy().reshape(-1, 3)

print(predictions[1])
print(true_values[1])

# 初始化一个变量来统计预测正确的数量
correct_predictions = 0

# 顺便保存1-norm值的数据
predictions_l1_norm = []
true_values_l1_norm = []
for i in range(len(predictions)):
    prediction_l1_norm = np.linalg.norm(predictions[i])
    true_value_l1_norm = np.linalg.norm(true_values[i])

    if abs(prediction_l1_norm - true_value_l1_norm) < 0.000001:
        correct_predictions += 1

    predictions_l1_norm.append(prediction_l1_norm)
    true_values_l1_norm.append(true_value_l1_norm)

# 计算准确率
accuracy = correct_predictions / len(predictions)
print("Test Accuracy:", accuracy)

# 保存到mat
data_to_save = {
    'predictions': predictions[:2000],
    'true_values': true_values[:2000]
}

# Define the filename for the .mat file
output_filename = 'predictions_and_true_values.mat'

# Save the dictionary containing predictions and true values in the .mat file
sio.savemat(output_filename, data_to_save)
