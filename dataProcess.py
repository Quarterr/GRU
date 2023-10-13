import numpy as np
import scipy.io as sio


# 创建一个新的MAT文件
new_mat_data = {}
for k in range(1,3):
    file_path = 'C:/Users/HP/Desktop/CEG5003/dataset/Dataset_nurbs_{}/training_set_transformer.mat'.format(k)
    mat_data = sio.loadmat(file_path)

    # 2. 对每个2000个一组的数据进行求导，并保存到新的MAT文件
    for i in range((k-1)*200+1, k*200+1):
        key = f'series_training{i}'
        print(key)
        data = mat_data[key]
        time_step = 0.01

        x = data[:, 0]
        x_1 = np.diff(x) / time_step
        x_1 = np.insert(x_1, 0, 0)
        x_2 = np.diff(x_1) / time_step
        x_2 = np.insert(x_2, 0, 0)
        x_3 = np.diff(x_2) / time_step
        x_3 = np.insert(x_3, 0, 0)

        # 获取第四列数据作为标签
        label_x = data[:, 3]

        # 将原始数据、一阶导数和二阶导数结果保存到新的MAT文件
        new_mat_data[key] = np.column_stack((x, x_1, x_2, x_3, label_x))

# 3. 保存新的MAT文件
sio.savemat('new_data.mat', new_mat_data)
