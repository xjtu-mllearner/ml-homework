from sklearn.preprocessing import OneHotEncoder
import pandas as pd
# Load the data with gbk encoding
file_path = r"C:\Users\李绪泰\OneDrive\桌面\链家数据.csv"
data = pd.read_csv(file_path, encoding='gbk')
# Check the unique values for each categorical feature

categorical_features = data[ ['所在楼层', '户型结构', '建筑类型', '房屋朝向', '建筑结构', '装修情况', '梯户比例', '供暖方式', '配备电梯', '房屋年限', '产权所属',]]
unique_values = {}
for feature in categorical_features.columns:
    unique_values[feature] = categorical_features[feature].unique()



# Define the categorical features for one-hot encoding
categorical_features = ['所在楼层', '户型结构', '建筑类型', '房屋朝向', '建筑结构', '装修情况', '梯户比例', '供暖方式', '配备电梯', '房屋年限', '产权所属', ]

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Encode the categorical features
encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
encoded_data.columns = ['所在楼层_' + str(cat) for cat in unique_values['所在楼层']] + \
                      ['户型结构_' + str(cat) for cat in unique_values['户型结构']] + \
                      ['建筑类型_' + str(cat) for cat in unique_values['建筑类型']] + \
                      ['房屋朝向_' + str(cat) for cat in unique_values['房屋朝向']] + \
                      ['建筑结构_' + str(cat) for cat in unique_values['建筑结构']] + \
                      ['装修情况_' + str(cat) for cat in unique_values['装修情况']] + \
                      ['梯户比例_' + str(cat) for cat in unique_values['梯户比例']] + \
                      ['供暖方式_' + str(cat) for cat in unique_values['供暖方式']] + \
                      ['配备电梯_' + str(cat) for cat in unique_values['配备电梯']] + \
                      ['房屋年限_' + str(cat) for cat in unique_values['房屋年限']] + \
                      ['产权所属_' + str(cat) for cat in unique_values['产权所属']]

# Merge the encoded data with the original data
merged_data = pd.concat([data.drop(categorical_features, axis=1), encoded_data], axis=1)

print(merged_data.head())
merged_data.to_csv('one_hot.csv',index=False)