import pandas as pd

# Load the CSV file into a DataFrame to inspect its content
file_path = r"C:\Users\李绪泰\OneDrive\桌面\链家数据.csv"
lianjia_data = pd.read_csv(file_path,encoding='gbk')

# Display the first few rows of the dataframe to understand its structure and content
print(lianjia_data.head())







# Perform data cleaning

# Check for missing values in the dataset
missing_values = lianjia_data.isnull().sum()

# Check data types of each column to ensure they are appropriate
data_types = lianjia_data.dtypes

# Display missing values and data types
print(missing_values, data_types)

# Remove non-numeric characters from '建筑面积' and convert to float
lianjia_data['建筑面积'] = lianjia_data['建筑面积'].str.replace(r'[^0-9.]', '', regex=True).astype(float)

# Now, check data types again to confirm the changes
data_types_after_conversion = lianjia_data.dtypes
print(data_types_after_conversion)







import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# Set the style of seaborn plots
sns.set_style("whitegrid")

# Plot the distribution of the '单价' (Price Per SQM) column
plt.figure(figsize=(10, 6))
sns.histplot(lianjia_data['单价'], bins=30, kde=True)
plt.title('单价 (Price Per SQM) Distribution')
plt.xlabel('单价 (Price Per SQM)')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of the '总价(万元)' (Total Price in Millions) column
plt.figure(figsize=(10, 6))
sns.histplot(lianjia_data['总价(万元)'], bins=30, kde=True)
plt.title('总价(万元) (Total Price in Millions) Distribution')
plt.xlabel('总价(万元) (Total Price in Millions)')
plt.ylabel('Frequency')
plt.show()



