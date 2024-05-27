#!/usr/bin/env python
# coding: utf-8

# In[53]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


class SVM_ADMM_hinge:
    def __init__(self, C, rho, max_iter=100, tol=1e-4):
        self.C = C
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
        self.z = None
        self.mu = None

    def admm_update(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        # 保存旧值
        w_old = self.w.copy()
        b_old = self.b
        z_old = self.z.copy()
        mu_old = self.mu.copy()

        # Update w
        self.w = np.linalg.solve((n_features * np.eye(n_features) + self.rho * X.T @ X),
                                 X.T @ (self.z - self.mu) - self.rho * self.w)

        # Update b
        # 注意：这里的更新可能需要根据实际问题进行调整
        self.b = (y_ - X @ self.w - self.z + self.mu).mean()

        # Update z
        self.z = np.maximum(0, 1 - y_ * (X @ self.w + self.b) + self.mu / self.rho)

        # Update mu
        self.mu += self.rho * (z_old - self.z)

        # 检查收敛性
        w_converged = np.linalg.norm(self.w - w_old) < self.tol
        b_converged = np.abs(self.b - b_old) < self.tol
        z_converged = np.linalg.norm(self.z - z_old) < self.tol
        mu_converged = np.linalg.norm(self.mu - mu_old) < self.tol

        # 返回所有变量是否收敛
        return w_converged and b_converged and z_converged and mu_converged

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0
        self.z = np.zeros(n_samples)
        self.mu = np.zeros(n_samples)

        for k in range(self.max_iter):
            if self.admm_update(X, y):
                break

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        print(np.mean(linear_output))
        return linear_output<np.mean(linear_output)

# 绘制ADMM SVM的混淆矩阵
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
# 加载鸢尾花数据集
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
X = data.data
y = data.target
# 只选择前两个特征和前两个类别
X = X[y < 2, :10]
y = y[y < 2]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练SVM模型
svm_admm = SVM_ADMM_hinge(C=1.0, rho=0.01)
svm_admm.fit(X_train, y_train)

# 进行预测
y_predictions = svm_admm.predict(X_test)
print(y_predictions)
# 评估模型
print(classification_report(y_test, y_predictions))
# 绘制数据点
plt.scatter(X_test[y_predictions == 0][:, 0], X_test[y_predictions == 0][:, 1], color='red', marker='o', label='Setosa')
plt.scatter(X_test[y_predictions == 1][:, 0], X_test[y_predictions == 1][:, 1], color='blue', marker='x', label='Versicolor')

# 设置图表属性
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM ADMM Decision Boundary')
plt.legend()
plt.show()




# 训练SVM模型
svm = SVC(C=2.0, random_state=1)
svm.fit(X_train, y_train)

# 进行预测
y_predictions_ = svm.predict(X_test)

# 评估模型
print(classification_report(y_test, y_predictions_))
# 绘制数据点
plt.scatter(X_test[y_predictions_ == 0][:, 0], X_test[y_predictions_ == 0][:, 1], color='red', marker='o', label='Setosa')
plt.scatter(X_test[y_predictions_ == 1][:, 0], X_test[y_predictions_ == 1][:, 1], color='blue', marker='x', label='Versicolor')

# 设置图表属性
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVC')
plt.legend()
plt.show()



# 使用ADMM SVM模型的预测结果绘制混淆矩阵
plot_confusion_matrix(y_test, y_predictions, 'ADMM SVM Confusion Matrix')

# 使用sklearn的SVM模型进行预测
y_predictions_sklearn = svm.predict(X_test)

# 绘制sklearn SVM的混淆矩阵
plot_confusion_matrix(y_test, y_predictions_sklearn, 'Sklearn SVM Confusion Matrix')

# 比较两者的分类报告
print("ADMM SVM Classification Report:")
print(classification_report(y_test, y_predictions))

print("Sklearn SVM Classification Report:")
print(classification_report(y_test, y_predictions_sklearn))


# In[ ]:




