import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


file_path = '链家数据.xlsx'

data = pd.read_excel(file_path)
# 删除包含缺失值的行
data.dropna(inplace=True)

numerical_stats = data.describe()
print(numerical_stats)

# Convert the '挂牌时间' column to datetime format
data['挂牌时间'] = pd.to_datetime(data['挂牌时间'])

# Sort the data by '挂牌时间' for a better visualization
data_sorted = data.sort_values(by='挂牌时间')

# Now let's plot the '单价' and '总价(万元)' over time
plt.figure(figsize=(15, 5))

# 单价
plt.subplot(1, 2, 1)
plt.scatter(data_sorted['挂牌时间'], data_sorted['单价'], color='blue', alpha=0.5)
plt.title('单价随挂牌时间的变化')
plt.xlabel('挂牌时间')
plt.ylabel('单价 (元/平方米)')

# 总价(万元)
plt.subplot(1, 2, 2)
plt.scatter(data_sorted['挂牌时间'], data_sorted['总价(万元)'], color='red', alpha=0.5)
plt.title('总价(万元)随挂牌时间的变化')
plt.xlabel('挂牌时间')
plt.ylabel('总价 (万元)')


plt.tight_layout()

# Display the plot
plt.show()


# 单价
plt.figure(figsize=(12, 6))
sns.histplot(data=data, x="单价", bins=50, kde=True, color="skyblue")
plt.title('单价分布')
plt.xlabel('单价')
plt.ylabel('频率')
plt.show()

# 总价
plt.figure(figsize=(12, 6))
sns.histplot(data=data, x="总价(万元)", bins=50, kde=True, color="salmon")
plt.title('总价分布')
plt.xlabel('总价')
plt.ylabel('频率')
plt.show()


sns.set(context='notebook',
style ='whitegrid',
font ='SimHei',
color_codes = True,
rc = None)


# Plot the boxplot of housing price per square meter grouped by district (区)
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x="区", y="单价")
plt.title('按区域划分的单价箱线图')
plt.xlabel('区')
plt.ylabel('单价')
plt.xticks(rotation=45)
plt.show()

# Plot the boxplot of housing price per square meter grouped by neighborhood (小区)
top_neighborhoods = data['小区'].value_counts().head(10).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[data['小区'].isin(top_neighborhoods)], x="小区", y="单价")
plt.title('按小区划分的单价箱线图(Top 10)')
plt.xlabel('小区')
plt.ylabel('单价')
plt.xticks(rotation=20)
plt.show()


# Perform one-hot encoding on the specified columns
one_hot_encoded_data = pd.get_dummies(data, columns=['房屋年限', '产权所属', '装修情况'])

# Display the first few rows of the dataframe with one-hot encoded columns
print(one_hot_encoded_data.head())


plt.figure(figsize=(10, 8))
sns.heatmap(one_hot_encoded_data [['建筑面积','总价(万元)','单价','室','厅','厨','卫','总楼层']].corr(), annot=True, cmap='coolwarm')
plt.title('热力图')
plt.show()

# Select the specified columns for scatter plots
selected_columns = ['建筑面积', '单价', '室']


# Define a function to plot scatter plots for each pair of columns
def plot_scatter_matrix(df, columns):
    fig, axes = plt.subplots(len(columns), len(columns), figsize=(20, 20))
    fig.suptitle('Scatter Matrix of Selected Columns', y=1.02)

    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            ax = axes[i, j]
            # Diagonal plots are treated differently (histograms)
            if i == j:
                sns.histplot(df[col_i], kde=True, ax=ax)
            else:
                sns.scatterplot(x=df[col_i], y=df[col_j], ax=ax)
                # Calculate and display the Pearson correlation coefficient
                corr = df[[col_i, col_j]].corr().iloc[0, 1]
                ax.annotate(f'ρ = {corr:.2f}', xy=(0.05, 0.9), xycoords=ax.transAxes, color='black')

            # Remove axis labels for clarity, except for the leftmost and bottommost plots
            if i != len(columns) - 1:
                ax.set_xlabel('')
            if j != 0:
                ax.set_ylabel('')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Call the function to plot the scatter matrix
plot_scatter_matrix(one_hot_encoded_data, selected_columns)