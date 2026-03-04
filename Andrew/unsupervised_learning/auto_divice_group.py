import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# ===================== 【强制】中文乱码永久修复配置 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ========================================================================

# ==========================================
# 1. 准备数据：生成人造聚类数据（无监督学习，我们假装不知道标签）
# ==========================================
print("正在生成聚类数据...")
# 生成300个样本，分成4堆，特征维度为2（方便可视化）
# 注意：我们生成了y_true，但无监督学习中完全不会使用它！
X, y_true = make_blobs(
    n_samples=300,    # 300个数据点
    n_features=2,     # 2个特征（x轴和y轴）
    centers=4,        # 真实的4个中心点
    cluster_std=0.60, # 每堆数据的离散程度
    random_state=0    # 固定随机种子，保证结果可复现
)

# ==========================================
# 2. 无监督学习核心：K-Means聚类
# ==========================================
print("正在进行K-Means无监督聚类...")
# 初始化K-Means模型：假设我们要分成4组（n_clusters=4）
kmeans = KMeans(n_clusters=4, random_state=0)

# 【关键】无监督学习：只输入特征X，完全不输入标签y_true！
kmeans.fit(X)

# 获取聚类结果：
# 1. 每个数据点被自动分配的组号（0, 1, 2, 3）
y_pred = kmeans.labels_
# 2. 算法自动找到的4个聚类中心
centers = kmeans.cluster_centers_

print("无监督聚类完成！")
print(f"算法自动找到了 {len(centers)} 个聚类中心。")

# ==========================================
# 3. 可视化结果（直观感受无监督学习的效果）
# ==========================================
plt.figure(figsize=(14, 6))

# 图1：原始数据（假装我们不知道它是分成4堆的）
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', edgecolors='k', alpha=0.6)
plt.title("原始数据（无监督学习视角：只有一堆点，没有标签）")
plt.xlabel("特征 1")
plt.ylabel("特征 2")

# 图2：K-Means无监督聚类结果
plt.subplot(1, 2, 2)
# 画出所有数据点，颜色根据算法自动分配的组号区分
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', edgecolors='k', alpha=0.6, label='数据点')
# 画出算法自动找到的聚类中心（红色大X）
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', linewidths=2, label='自动找到的聚类中心')
plt.title("K-Means无监督聚类结果（算法自动分组）")
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 4. 结果解读
# ==========================================
print("\n=== 无监督学习结果解读 ===")
print("1. 我们没有给算法任何标签（没告诉它哪堆是哪堆）。")
print("2. 算法只看数据点的位置（特征），自动把相似的点分成了4组。")
print("3. 红色的X是算法自动计算出的每个组的“中心点”。")
print("4. 这就是无监督学习：从无标签的数据中，自动发现内在结构。")