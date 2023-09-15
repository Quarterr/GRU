import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 从MAT文件中加载数据
mat_data = sio.loadmat('C:/Users/HP/Desktop/CEG5003/dataset/Dataset_nurbs_1/training_set_transformer.mat')


# 创建一个新的MAT文件
new_mat_data = {}

# 2. 对每个2000个一组的数据进行求导，并保存到新的MAT文件
for i in range(1, 21):
    key = f'series_training{i}'
    data = mat_data[key]

    # 获取第一列数据
    first_column = data[:, 0]  # 假设第一列是你要的数据

    # 计算导数
    time_step = 0.01
    derivative_data = np.diff(first_column) / time_step
    derivative_data = np.insert(derivative_data, 0, 0)
    second_derivative_data = np.diff(derivative_data) / time_step
    second_derivative_data = np.insert(second_derivative_data, 0, 0)
    third_derivative_data = np.diff(second_derivative_data) / time_step
    third_derivative_data = np.insert(third_derivative_data, 0, 0)

    # 获取第四列数据作为标签
    label = data[:, 3]

    # 将原始数据、一阶导数和二阶导数结果保存到新的MAT文件
    new_mat_data[key] = np.column_stack((data, derivative_data, second_derivative_data, third_derivative_data, label))

# 3. 保存新的MAT文件
sio.savemat('new_data.mat', new_mat_data)

# 加载新的MAT文件
new_mat_data = sio.loadmat('new_data.mat')

# 获取第一列数据和标签
series1_data = new_mat_data['series_training1']
first_column = series1_data[:, 0]
label = series1_data[:, -1]  # 假设标签位于最后一列


# 绘制图表，将第一列数据作为纵坐标，标签作为横坐标
plt.figure(figsize=(10, 6))
plt.plot(first_column, label)
plt.xlabel('input')
plt.ylabel('output')
plt.grid(True)

# 显示图表
plt.show()

