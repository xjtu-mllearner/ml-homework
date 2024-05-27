#!/usr/bin/env python
# coding: utf-8

# In[2]:


from random import choice, random
from numpy import array, ones, zeros, ndarray, where, isnan, inf, intersect1d, maximum
import matplotlib.pyplot as plt
from typing import Sequence
from numpy import ndarray, maximum, zeros_like
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
import time


class GradientDescent:
    """梯度下降法"""
    def __init__(self,
             θ__: ndarray,          # ndarray类型的向量或矩阵：待优化的参数
             LR: float = 0.01,      # 全局学习率
             method: str = 'Adam',  
             decayRates_: Sequence[float] = (.9, .999),  # 两个衰减率
             ):
        assert type(θ__)==ndarray and θ__.ndim>=1, '待优化的参数θ__的数据类型应为ndarray，其属性ndim应不小于1'
        assert len(decayRates_)==2, '衰减率参数decayRates_长度应为2'
        assert 0<=decayRates_[0]<1, '首项衰减率的取值范围为[0, 1)'
        assert 0<=decayRates_[1]<1, '次项衰减率的取值范围为[0, 1)'
        self.θ__ = θ__                  # ndarray类型的向量或矩阵：待优化的参数
        self.LR = LR                    # 全局学习率
        self.β1, self.β2 = decayRates_  # 衰减率参数（第一衰减率、第二衰减率）
        self.Δθ__ = zeros_like(θ__)     # 向量或矩阵：参数更新的增量
        # 学习率调整策略，可选'SGD'/'AdaGrad'/'RMSprop'/'AdaDelta'/'Momentum'/'Adam'/'Nesterov'/'AdaMax'/'Nadam'/'NadaMax'
        self.method = method.lower()
        self.m__ = zeros_like(θ__)      # 调整学习率所用到的累计参数
        self.n__ = zeros_like(θ__)      # 调整学习率所用到的累计参数
        self.t = 0                      # 更新的次数
        self.optmizer = self.Adam


    def update(self, grad__: ndarray) -> ndarray:
        """输入参数的梯度（向量或矩阵），更新参数"""
        self.t += 1            # 更新的次数
        self.optmizer(grad__)  # 得到参数更新的增量self.Δθ__
        self.θ__ += self.Δθ__  # 更新参数
        return self.θ__

    def Adam(self, grad__):
        t = self.t     # 读取：更新的次数
        β1 = self.β1   # 读取：衰减率
        β2 = self.β2   # 读取：衰减率
        self.m__[:] = β1*self.m__ + (1 - β1)*grad__     # 更新“有偏一阶矩估计”（用于Momentum）
        self.n__[:] = β2*self.n__ + (1 - β2)*grad__**2  # 更新“有偏二阶原始矩估计”（用于RMSprop）
        self.Δθ__[:] = -self.LR*(1 - β2**t)**0.5/(1 - β1**t) * self.m__/(self.n__**0.5 + 1e-8)  # 参数更新的增量

class LinearKernel:
    """
    线性核函数（Linear kernel function）
    核函数值 K(x_, z_) = x_ @ z_
    """
    def __call__(self, X__: ndarray, Z__: ndarray) -> ndarray:
        if X__.ndim==1 and 1<=Z__.ndim<=2:     # 若X__为1维ndarray，且Z__为1维或2维ndarray
            return Z__ @ X__
        elif 1<=X__.ndim<=2 and Z__.ndim==1:   # 若Z__为1维ndarray，且X__为1维或2维ndarray
            return X__ @ Z__
        elif X__.ndim==2 and Z__.ndim==2:      # 若X__、Z__均为2维ndarray
            K__ = zeros([len(X__), len(Z__)])  # 核函数值矩阵
            for n, x_ in enumerate(X__):
                K__[n] = Z__ @ x_
            return K__
        else:
            raise ValueError('输入量X__、Z__应为ndarray，其属性ndim应为1或2')

class SupportVectorMachineClassification:
    """支持向量机分类"""
    def __init__(self,
            C: float = 1.,               # 超参数：惩罚参数
            kernel: str = 'linear',      # 核函数： 线性核'linear'
            solver: str = 'Pegasos',        
            maxIterations: int = 50000,  # 最大迭代次数
            LR: float = 0.1,    # 超参数：全局学习率（用于Pegasos求解算法）
            ):

        self.C = C                    # 超参数：惩罚参数
        self.kernel = kernel.lower()  # 核函数：线性核'linear
        self.solver = solver.lower()  # 求解算法：梯度下降'Pegasos'
        self.maxIterations = maxIterations  # 最大迭代次数
        self.LR = LR    # 超参数：全局学习率（用于Pegasos求解算法）
        self.M = None   # 输入特征向量的维数
        self.w_ = None  # M维向量：权重向量
        self.b = None   # 偏置
        self.α_ = None  # N维向量：所有N个训练样本的拉格朗日乘子
        self.supportVectors__ = None  # 矩阵：所有支持向量
        self.αSV_ = None     # 向量：所有支持向量对应的拉格朗日乘子α
        self.ySV_ = None     # 向量：所有支持向量对应的标签
        self.minimizedObjectiveValues_ = None  # 列表：历次迭代的最小化目标函数值（对于Pegasos算法，指损失函数值；对于SMO求解算法，指对偶问题的最小化目标函数值）



    def fit(self, X__: ndarray, y_: ndarray):
        assert type(X__)==ndarray and X__.ndim==2, '输入训练样本矩阵X__应为2维ndarray'
        assert type(y_)==ndarray  and y_.ndim==1, '输入训练标签y_应为1维ndarray'
        assert len(X__)==len(y_), '输入训练样本数量应等于标签数量'
        assert set(y_)=={-1, 1},  '输入训练标签取值应为-1或+1'
        self.M = X__.shape[1]    # 输入特征向量的维数
        self.Pegasos(X__, y_)
        return self

    def Pegasos(self, X__, y_):
        """使用Pegasos（Primal estimated sub-gradient solver）算法，即梯度下降法，求解原始优化问题，使损失函数值最小化"""
        C = self.C        # 读取：惩罚参数
        N, M = X__.shape  # 训练样本数量N、输入特征向量的维数
        w_ = ones(M)      # M维向量：初始化权重向量
        b = array([0.])   # 初始化偏置（由于偏置b需要载入梯度下降优化器，故先定义为1维ndarray，优化结束后再将偏置b提取为浮点数float）
        optimizer_for_w_ = GradientDescent(w_, method='Adam', LR=self.LR)  # 实例化w_的梯度下降优化器，代入全局学习率LR，选择Adam学习率调整策略
        optimizer_for_b  = GradientDescent(b, method='Adam', LR=self.LR)   # 实例化b的梯度下降优化器，代入全局学习率LR，选择Adam学习率调整策略
        minLoss = inf     # 初始化最小损失函数值
        self.minimizedObjectiveValues_ = losses_ = []  # 列表：记录每一次迭代的损失函数值
        for t in range(1, self.maxIterations + 1):
            ξ_ = maximum(1 - y_*(X__ @ w_ + b), 0)  # N维向量：N个松弛变量
            I_ = ξ_>0                        # N维向量：N个0/1指示量
            loss = 0.5*w_ @ w_ + C*ξ_.sum()  # 损失函数值
            gradw_ = w_ - C*(I_*y_ @ X__)    # M维向量：损失函数对权重向量w_的梯度
            gradb = -C*(I_ @ y_)             # 损失函数对偏置b的梯度
            losses_.append(loss)             # 记录损失函数值
            if loss<minLoss:
                minLoss = loss         # 记录历史最小损失函数
                wOptimal_ = w_.copy()  # 记录历史最优权重向量w_
                bOptimal = b.copy()    # 记录历史最优偏置b
            print(f'第{t}次迭代，损失函数值{loss:.5g}')
            # 梯度下降，更新权重向量w_和偏置b
            optimizer_for_w_.update(gradw_)  # 代入梯度至优化器，更新权重向量w_
            optimizer_for_b.update(gradb)    # 代入梯度至优化器，更新偏置b
        else:
            print(f'达到最大迭代次数{self.maxIterations}\n############################')

        self.w_ = wOptimal_.copy()      # 以历史最优权重向量作为训练结果
        self.b = bOptimal.copy().item() # 以历史最优偏置作为训练结果

   
    def predict(self, X__: ndarray) -> ndarray:
        """测试"""
        assert type(X__)==ndarray and X__.ndim==2, '输入测试样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        y_ = where(self.decisionFunction(X__)>=0, 1, -1)  # 判定类别
        return y_

    def decisionFunction(self, X__: ndarray) -> ndarray:
        """计算决策函数值 f = w_ @ x_ + b"""
        assert X__.ndim==2, '输入样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        f_ = X__ @ self.w_ + self.b
        return f_

    def accuracy(self, X__: ndarray, y_: ndarray) -> float:
        """计算测试正确率"""
        测试样本正确个数 = sum(self.predict(X__)==y_)
        测试样本总数 = len(y_)
        return 测试样本正确个数/测试样本总数


def plot_confusion_matrices(cm1, cm2, labels, title1="Confusion Matrix own", title2="Confusion Matrix of SVC"):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrices Comparison')

    # 绘制第一个混淆矩阵
    ax1.imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title(title1)
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    thresh = cm1.max() / 2.
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            ax1.text(j, i, format(cm1[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm1[i, j] > thresh else "black")

    # 绘制第二个混淆矩阵
    ax2.imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title(title2)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    thresh = cm2.max() / 2.
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            ax2.text(j, i, format(cm2[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm2[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # 设置随机种子
    from numpy.random import seed; seed(0)
    from random import seed; seed(0)


    from sklearn.model_selection import train_test_split
    data=load_breast_cancer()
    x=data.data
    y=data.target
    y[y == 0] = -1
    Xtrain__, Xtest__, ytrain_, ytest_ = train_test_split(x, y)

    # 实例化支持向量机分类模型
    model = SupportVectorMachineClassification(
        C=1.,             # 超参数：惩罚参数
        kernel='linear',  # 核函数：线性核'linear'
        solver='Pegasos',     # 求解算法：梯度下降法'Pegasos'
        maxIterations=50000,  # 最大迭代次数
        LR=0.01,   # 超参数：全局学习率（用于Pegasos求解算法）
        )
    start_time = time.time()
    model.fit(Xtrain__, ytrain_)        # 训练
    end_time = time.time()  # 算法结束后记录时间
    print(f"算法运行时间：{end_time - start_time} 秒")
    print(f'权重向量w_ = {model.w_}')
    print(f'偏置b = {model.b}')
    print(f'训练集正确率：{model.accuracy(Xtrain__, ytrain_):.3f}')
    print(f'测试集正确率：{model.accuracy(Xtest__, ytest_):.3f}')
   
    
    
    # 对比sklearn的支持向量机分类
    print('\n使用同样的超参数、训练集和测试集，对比sklearn的支持向量机分类：')
    from sklearn.svm import SVC
    modelSK = SVC(
        C=model.C,            # 超参数：惩罚参数
        kernel=model.kernel,  # 核函数
        max_iter=model.maxIterations, # 最大迭代次数
        )
    start_time = time.time()
    modelSK.fit(Xtrain__, ytrain_)  # 训练
    end_time = time.time()  # 算法结束后记录时间
    print(f"算法运行时间：{end_time - start_time} 秒")
    print(f'sklearn的权重向量w_ = {modelSK.coef_[0]}')
    print(f'sklearn的偏置b = {modelSK.intercept_[0]}')
    print(f'sklearn的训练集正确率：{modelSK.score(Xtrain__, ytrain_):.3f}')
    print(f'sklearn的测试集正确率：{modelSK.score(Xtest__, ytest_):.3f}')
    
    y_pred_own = model.predict(Xtest__)
    y_pred_sklearn = modelSK.predict(Xtest__)
    
    cm_own = confusion_matrix(ytest_, y_pred_own)
    cm_sklearn = confusion_matrix(ytest_, y_pred_sklearn)
    plot_confusion_matrices(cm_own, cm_sklearn, ['-1','1'])


# In[ ]:





# In[ ]:




