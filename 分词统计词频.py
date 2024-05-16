import jieba
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# 指定默认字体，解决保存时中文字符显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取链家数据CSV文件
file_path = r"C:\Users\李绪泰\OneDrive\桌面\链家数据.csv"
df = pd.read_csv(file_path, encoding='gbk')

# 读取停用词表
with open(r"C:\Users\李绪泰\OneDrive\桌面\停用词.txt", 'r', encoding='utf-8') as f:
    stopwords = set(f.read().split('\n'))

# 使用jieba分词并统计词频，同时去除停用词
word_list = [word for title in df['标题'] for word in jieba.cut(title) if word not in stopwords and word.strip()]
word_freq = Counter(word_list)
print(word_list)
# 过滤低频词，只保留词频大于等于100的词语
common_words = {word: freq for word, freq in word_freq.items() if freq >= 30}

# 按词频降序排序常见词
sorted_common_words = dict(sorted(common_words.items(), key=lambda item: item[1], reverse=True))

# 绘制词频图
plt.figure(figsize=(8, 4))
plt.bar(sorted_common_words.keys(), sorted_common_words.values(),width=0.8)
plt.title('词频图', fontsize=15)
plt.xlabel('词语', fontsize=1)
plt.ylabel('频率', fontsize=12)
plt.xticks(rotation=60, fontsize=10)
plt.tight_layout()
plt.show()
