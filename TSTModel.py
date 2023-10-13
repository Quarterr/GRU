from torch.optim.lr_scheduler import StepLR
import numpy as np
import scipy.io as sio
import dataPreprocess
import torch
import torch.nn as nn
import math


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


def generate_positional_encoding(input_dim, hidden_dim, dropout):
    pe = torch.zeros(1, input_dim, hidden_dim)
    position = torch.arange(0, input_dim, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim))
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return nn.Parameter(pe, requires_grad=False)


def generate_mask(seq_length):
    mask = torch.triu(torch.ones(seq_length, seq_length) == 1, diagonal=1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_heads, hidden_dim, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = generate_positional_encoding(9, hidden_dim, dropout)  # 固定时间步数为9

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input embedding
        x = self.embedding(x)

        # Get the sequence length (number of time steps)
        seq_length = x.size(1)

        # Add positional encoding
        x += self.positional_encoding[:, :seq_length, :]

        # mask = generate_mask(seq_length)  # 生成掩码
        #
        # # Transformer encoder layers
        # for layer in self.transformer_layers:
        #     x = layer(x, src_key_padding_mask=mask)

        # Transformer decoder layers
        for layer in self.decoder_layers:
            x = layer(x, memory=x)  # 使用自身作为解码器的输入

        # Output layer
        x = self.output_layer(x)  # No need to select the last time step

        return x[:, -1, :]  # 最后一个时间步的输出


if __name__ == "__main__":
    # 获取训练集和测试集
    train_loader, test_loader = dataPreprocess.preprocess_data()

    # 检测gpu是否可用
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoding_layers = 2
    heads = 2
    hidden_dims = 16
    model = TimeSeriesTransformer(config.feature_size, config.output_size, encoding_layers, heads, hidden_dims).to(
        device)  # 定义GRU网络
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
            x_train = x_train.to(device)  # 300, 9, 4
            y_train = y_train.to(device)
            optimizer.zero_grad()
            y_train_pred = model(x_train)
            loss = loss_function(y_train_pred, y_train.reshape(-1, config.output_size))
            loss.backward()
            optimizer.step()
            if loss < prev_loss:
                torch.save(model, 'tst.pt')
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
