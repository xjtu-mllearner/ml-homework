import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import time

# 参数设置
n_samples = 100000
n_features = 10
test_size = 0.2
k_values = range(1, 51)  # k值从1到50

# 生成数据集
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 初始化变量来存储结果
localization_times = []
boosting_times = []
mses = []

# 对每个k值进行操作
for k in k_values:
    start_time = time.time()

    # 初始化KNeighborsRegressor
    knn_reg = KNeighborsRegressor(n_neighbors=k)

    # 第一步：选择最接近测试点的k个样本
    knn_reg.fit(X_train, y_train)
    distances, indices = knn_reg.kneighbors(X_test)

    # 第二步：对局部样本进行平均处理
    X_train_local = np.array([X_train[idx] for idx in indices]).mean(axis=1)
    y_train_local = np.array([y_train[idx] for idx in indices]).mean(axis=1)

    # 记录局部化方法的时间
    localization_time = time.time() - start_time

    # 再用boosting算法处理
    start_time = time.time()
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train_local, y_train_local)

    # 记录boosting算法的时间代价
    boosting_time = time.time() - start_time

    # 评估模型性能
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 存储结果
    localization_times.append(localization_time)
    boosting_times.append(boosting_time)
    mses.append(mse)

    # 打印结果
    # print(f"k={k}, Localization time: {localization_time} seconds, Boosting time: {boosting_time} seconds, MSE: {mse}")

# 绘图
fig, ax1 = plt.subplots(figsize=(10, 6))
# 调整右侧边界，使其更靠近图形的左侧，从而在图形右侧留下更多的空白。
# 值越小，右侧空白越大。默认值为1。
fig.subplots_adjust(right=0.8)  # 例如，将右边界设置为整个图形宽度的80%

# 绘制mse
color = 'tab:blue'
ax1.set_xlabel('k value')
ax1.set_ylabel('MSE', color=color)
ax1.plot(k_values, mses, label='MSE', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个y轴
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Localization Time', color=color)  # 第二个y轴标签
ax2.plot(k_values, localization_times, label='Localization Time', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# 创建第三个Y轴
ax3 = ax1.twinx()
color = 'tab:green'
# 调整ax3位置，为了不与ax2重叠
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('Boosting Time', color=color)  # 第三个y轴标签
ax3.plot(k_values, boosting_times, label='Boosting Time', color=color)  # 第三个y轴的数据
ax3.tick_params(axis='y', labelcolor=color)


plt.title('Effect of k on Localization Time, Boosting Time, and MSE')
plt.grid(True)
plt.show()
