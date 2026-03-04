import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 3. 加载MNIST数据集（对应教材"经验E"）
print("正在加载MNIST数据集...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)  # 标签转为整数

# 4. 数据预处理（归一化+划分训练/测试集）
X = X / 255.0  # 像素值归一化到[0,1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 80%训练，20%测试
)

# 5. 定义学习器（对应教材"假设空间H"）
# 归纳偏置：欧氏距离度量 + k=3近邻投票
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# 6. 训练模型（对应教材"通过经验E改进性能P"）
print("正在训练k近邻分类器...")
knn.fit(X_train, y_train)

# 7. 评估性能（对应教材"性能指标P"）
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {accuracy:.4f}")

# 8. 可视化错误案例（分析归纳偏置的局限性）
errors = np.where(y_pred != y_test)[0]
plt.figure(figsize=(10, 5))
for i, idx in enumerate(errors[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"True:{y_test[idx]} Pred:{y_pred[idx]}")
    plt.axis('off')
plt.show()