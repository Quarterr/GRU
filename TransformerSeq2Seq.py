from torch.optim.lr_scheduler import StepLR
import numpy as np
import scipy.io as sio
import dataPreprocess
import torch
import torch.nn as nn


class Config:
    time_step = 9  # 时间步长，就是利用多少时间窗口
    feature_size = 4  # 每个步长对应的特征数量
    output_size = 1  # 输出6维向量。3个位置和对应的导数

    batch_size = 32  # 批次大小
    epochs = 100  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.001  # 初始化学习率
    model_name = 'tst'  # 模型名称
    save_path = './{}.pth'.format(model_name)  # 最优模型保存路径
    dropout = 0.1
    n_layers = 3 #编码器和解码器的层数
    n_heads = 2 #多头注意力头数
    # 定义一些超参数
    d_model = 26  # 模型维度
    d_ff = 256  # 前馈网络的维度


config = Config()




# 定义transformer模型类
class Transformer(nn.Module):
    def __init__(self, n_features, n_heads, d_model, d_ff, n_layers, dropout):
        super(Transformer, self).__init__()
        self.n_features = n_features  # 特征数量
        self.n_heads = n_heads  # 多头注意力的头数
        self.d_model = d_model  # 模型维度
        self.d_ff = d_ff  # 前馈网络的维度
        self.n_layers = n_layers  # 编码器和解码器的层数
        self.dropout = dropout  # dropout比率

        # 定义编码器层，包含多头自注意力，前馈网络，残差连接和层归一化
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
        # 定义编码器，包含n_layers个编码器层
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        # 定义解码器层，包含多头自注意力，多头编码器-解码器注意力，前馈网络，残差连接和层归一化
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
        # 定义解码器，包含n_layers个解码器层
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers)
        # 定义线性层，将模型输出映射到一个值，作为预测结果
        self.linear = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        # src: 输入序列，形状为(time_step, batch_size, n_features)
        # tgt: 目标序列，形状为(seq_len, batch_size, n_features)
        # 输出: 预测序列，形状为(seq_len, batch_size, 1)

        # 将输入序列和目标序列转换为模型维度
        src = src.view(config.time_step, config.batch_size, -1) * np.sqrt(self.d_model)
        tgt = tgt.view(config.time_step, config.batch_size, -1) * np.sqrt(self.d_model)

        # 对输入序列和目标序列进行编码器和解码器的处理，得到输出序列
        output = self.decoder(tgt, self.encoder(src))

        # 对输出序列进行线性层的映射，得到预测序列
        output = self.linear(output)

        return output



if __name__ == "__main__":
    # 获取训练集和测试集
    train_loader, test_loader = dataPreprocess.preprocess_data(1)

    # 检测gpu是否可用
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型，学习率，损失函数，优化器  nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
    model = Transformer(config.feature_size, config.n_heads, config.d_model, config.d_ff, config.n_layers, config.dropout).to(device)
    loss_function = nn.L1Loss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # 定义优化器
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    # 7.模型训练 并且保存效果最优模型
    prev_loss = 1000

    for epoch in range(config.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        running_loss = 0.0
        for data in train_loader:
            x_train, y_train = data
            x_train = x_train.to(device)  # 300, 9, 4
            y_train = y_train.to(device)  # 300, 9, 1
            optimizer.zero_grad()
            y_train_pred = model(x_train, y_train)
            loss = loss_function(y_train_pred, y_train.reshape(-1, config.output_size))
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
