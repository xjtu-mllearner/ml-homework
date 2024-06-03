import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import time
import math

# 参数设置
n_samples = 100000  # 可以根据需要调整数据规模
n_features = 10
noise = 0.1
random_state = 42
bandwidths = np.arange(0.1, 1.0, 0.1)  # 带宽从0.1到0.9，步长为0.1

# 用于存储结果的列表
bandwidth_values, num_data_points, mses_init, mses_boosted, kernel_times, boosting_times = [], [], [], [], [], []

# 生成数据集
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# 对不同带宽进行操作
for bandwidth in bandwidths:
    # 核估计函数，选择代表性的数据点
    def kernel_density(X, y, bandwidth):
        distances = np.linalg.norm(X, axis=1)
        indices = np.argsort(distances)[:int(len(X) * bandwidth)]
        return X[indices], y[indices]


    # 核估计选择数据点
    X_train_kernel, y_train_kernel = kernel_density(X_train, y_train, bandwidth)

    # 记录时间代价
    start_time = time.time()

    # 使用核估计选择的数据点训练初始模型
    model_init = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)
    model_init.fit(X_train_kernel, y_train_kernel)

    # 记录核估计和模型训练时间
    kernel_and_training_time = time.time() - start_time

    # 应用boosting
    start_time = time.time()
    model_boosted = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                              random_state=random_state)
    model_boosted.fit(X_train, y_train)

    # 记录boosting时间
    boosting_time = time.time() - start_time

    # 评估模型性能
    y_pred_init = model_init.predict(X_test)
    y_pred_boosted = model_boosted.predict(X_test)
    mse_init = mean_squared_error(y_test, y_pred_init)
    mse_boosted = mean_squared_error(y_test, y_pred_boosted)

    # 存储结果
    bandwidth_values.append(bandwidth)
    num_data_points.append(len(X_train_kernel))
    mses_init.append(mse_init)
    mses_boosted.append(mse_boosted)
    kernel_times.append(kernel_and_training_time)
    boosting_times.append(boosting_time)

# 创建图形和主Y轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制第一个数据集 - 左侧的第一个Y轴
ax1.plot(bandwidth_values, num_data_points, 'b-')
ax1.set_xlabel('Bandwidth')
ax1.set_ylabel('Number of data points used', color='b')
ax1.tick_params('y', colors='b')

# 创建左侧的第二个Y轴
ax2 = ax1.twinx()
# 调整ax2的位置，为了不与ax1重叠
ax2.spines['left'].set_position(('outward', 60))
ax2.yaxis.set_ticks_position('left')
ax2.yaxis.set_label_position('left')
ax2.plot(bandwidth_values, mses_init, 'g-')
ax2.set_ylabel('MSE', color='g')
ax2.tick_params('y', colors='g')

# 创建右侧的第一个Y轴
ax3 = ax1.twinx()
ax3.plot(bandwidth_values, kernel_times, 'r-')
ax3.set_ylabel('Kernel Density and Training Time', color='r')
ax3.tick_params('y', colors='r')

# # 创建右侧的第二个Y轴
# ax4 = ax1.twinx()
# # 调整ax4的位置，为了不与ax3重叠
# ax4.spines['right'].set_position(('outward', 60))
# ax4.plot(bandwidth_values, boosting_times, 'm-')
# ax4.set_ylabel('Boosting Time', color='m')
# ax4.tick_params('y', colors='m')

# 调整图的边界，为了有更多的空白来展示所有的y轴
fig.subplots_adjust(left=0.2, right=0.8)

plt.show()
