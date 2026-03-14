# 1. 导入必要库
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 2. 加载经典数据集（鸢尾花分类）
# 数据集说明：3类鸢尾花，4个特征（花萼长度/宽度、花瓣长度/宽度）
iris = load_iris()
X = iris.data  # 特征数据（4列：花萼长、花萼宽、花瓣长、花瓣宽）
y = iris.target  # 标签（0/1/2 对应3种鸢尾花）
feature_names = iris.feature_names  # 特征名（方便可视化）
class_names = iris.target_names  # 类别名（方便可视化）

# 3. 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # random_state固定随机数，结果可复现
)

# 4. 构建并训练决策树模型
# 关键参数：
# - criterion='gini'：用基尼系数选特征（也可换'entropy'信息熵）
# - max_depth=3：限制树深度，防止过拟合（核心剪枝手段）
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# 5. 模型评估（看测试集准确率）
test_accuracy = dt_model.score(X_test, y_test)
print(f"决策树测试集准确率：{test_accuracy:.2f}")

# 6. 可视化决策树（最能体现决策树的核心！）
plt.figure(figsize=(12, 8))  # 设置图大小
# plot_tree参数说明：
# - feature_names：特征名（每个节点显示用哪个特征）
# - class_names：类别名（叶节点显示分类结果）
# - filled=True：填充颜色（按纯度区分）
# - rounded=True：圆角框
plot_tree(dt_model,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True)
plt.title("鸢尾花分类决策树", fontsize=16)
plt.show()

# 7. 单个样本预测（直观演示）
# 随便选一个测试样本（比如第0个）
sample = X_test[0].reshape(1, -1)  # 转成模型要求的形状（1行4列）
pred_label = dt_model.predict(sample)[0]
true_label = y_test[0]
print(f"\n单个样本预测：")
print(f"样本特征：{X_test[0]}")
print(f"预测类别：{class_names[pred_label]}")
print(f"真实类别：{class_names[true_label]}")