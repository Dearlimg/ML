# 导入必要库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ========== 关键：添加中文显示配置 ==========
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用微软雅黑显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练模型
model = DecisionTreeClassifier(max_depth=300)
model.fit(X_train, y_train)

# 4. 预测并评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率：{accuracy:.2f}")

# 5. 可视化
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('花萼长度')  # 中文标签
plt.ylabel('花萼宽度')  # 中文标签
plt.title('鸢尾花数据分布')  # 中文标题
plt.show()