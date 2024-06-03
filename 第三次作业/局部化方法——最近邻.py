#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import time
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 生成大规模数据集
X, y = make_regression(n_samples=100000, n_features=10, noise=0.1, random_state=42)

# 定义一个函数来同步缩减数据集和目标变量
def synchronize_data_and_target(X, y, n_neighbors):

  neigh = NearestNeighbors(n_neighbors=n_neighbors)
  neigh.fit(X)
  distances, indices = neigh.kneighbors(X)
  # 初始化缩减后的数据集和目标变量
  X_reduced = []
  y_reduced = []
  added_indices = set()
  for i, neighbor_indices in enumerate(indices):
    # 如果该点还没有被添加过
    if i not in added_indices:
      X_reduced.append(X[i])
      y_reduced.append(y[i])
      added_indices.update(neighbor_indices)

  return np.array(X_reduced), np.array(y_reduced)
num=[]
err=[]
time1=[]
time2=[]
for n in list(range(1, 101, 5)):
        # 同步缩减数据集和目标变量
        X_representative_sync, y_representative_sync = synchronize_data_and_target(X, y, n)
        # 再次将缩减后的数据集和目标变量分割成训练集和测试集
        X_train_sync, X_test_sync, y_train_sync, y_test_sync = train_test_split(X_representative_sync, y_representative_sync,
                                                                                test_size=0.2, random_state=42)
        # 初始化梯度提升树模型
        gbt_sync = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        # 记录训练时间
        start_time_train = time.time()
        # 训练模型
        gbt_sync.fit(X_train_sync, y_train_sync)
        end_time_train = time.time()
        start_time_test = time.time()
        # 使用训练好的模型进行预测
        y_pred_sync = gbt_sync.predict(X_test_sync)
        end_time_test = time.time()
        # 评估模型性能
        mse_sync = mean_squared_error(y_test_sync, y_pred_sync)
        print(f"最终数据量{X_representative_sync.shape[0]},MSE:{mse_sync},训练时间:{end_time_train-start_time_train},测试时间:{end_time_test-start_time_test}")
        num.append(X_representative_sync.shape[0])
        err.append(mse_sync)
        time1.append(end_time_train-start_time_train)
        time2.append(end_time_test-start_time_test)

data=pd.DataFrame()
data['数据量']=num
data['MSE']=err
data['训练时间']=time1
data['测试时间']=time2
print(data)


# 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))
# 绘制MSE，并添加标签
color = 'tab:blue'
ax1.set_xlabel('数据量')
ax1.set_ylabel('MSE', color=color)
ax1.plot(data['数据量'], data['MSE'], color=color, label='MSE')
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个坐标轴
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('训练时间', color=color)
ax2.plot(data['数据量'], data['训练时间'], color=color, label='训练时间')
ax2.tick_params(axis='y', labelcolor=color)

# 创建第三个坐标轴
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.05))  # 将第三个坐标轴移动到右边， but not too far
color = 'tab:green'
ax3.set_ylabel('测试时间', color=color)
ax3.plot(data['数据量'], data['测试时间'], color=color, label='测试时间')
ax3.tick_params(axis='y', labelcolor=color)

# 添加标题和图例
plt.title('数据量、MSE、训练时间和测试时间的可视化')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax3.legend(loc='lower right')

# 显示图表
plt.show()



# In[ ]:




