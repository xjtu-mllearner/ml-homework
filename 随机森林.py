import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 假设CSV文件名为"data.csv"，且包含列名为'zhangdiefu'的目标变量和其他特征列
file_path = '2024-04-24_stock_data.csv'
df = pd.read_csv(file_path)

# 将数据分为特征和目标变量
X = df.drop(['涨跌幅','股票代码','股票简称','行业'], axis=1)
y = df['涨跌幅']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
import matplotlib.pyplot as plt

# 假设y_test和y_pred是已经计算好的真实值和预测值
# 这里我们使用示例数据来绘制图表


# 绘制真实值和预测值
plt.scatter(y_test, y_pred, color='blue', label='预测值 vs 真实值')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('随机森林模型预测值与真实值对比')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()
