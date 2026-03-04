import numpy as np
import matplotlib.pyplot as plt

# ===================== 【永久生效】中文乱码修复核心配置 =====================
# 固定使用Windows系统自带的中文字体，无需额外安装
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
# 解决负号'-'显示为方块的兼容问题
plt.rcParams['axes.unicode_minus'] = False
# ============================================================================

# ==========================================
# 1. 准备数据（对应吴恩达课程癌症分类数据集）
# ==========================================
# 特征：肿瘤大小（单位：cm）
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
# 标签：0=良性, 1=恶性
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# ==========================================
# 2. 定义核心函数（对应吴恩达课程公式）
# ==========================================
def sigmoid(z):
    """
    Sigmoid函数：把任意实数映射到(0,1)区间，代表概率
    对应吴恩达课程：g(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    """
    计算交叉熵代价函数（逻辑回归的代价函数）
    对应吴恩达课程：J(w,b) = (-1/m) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
    """
    m = X.shape[0]
    z = w * X + b
    y_hat = sigmoid(z)
    # 加上极小值epsilon，避免log(0)报错
    epsilon = 1e-9
    cost = (-1 / m) * np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
    return cost

def gradient_descent_logistic(X, y, w_init, b_init, alpha, num_iters):
    """
    逻辑回归的梯度下降（形式上和线性回归一致，y_hat通过sigmoid计算）
    """
    m = X.shape[0]
    w = w_init
    b = b_init
    cost_history = []

    for i in range(num_iters):
        z = w * X + b
        y_hat = sigmoid(z)
        # 梯度计算
        dw = (1 / m) * np.sum((y_hat - y) * X)
        db = (1 / m) * np.sum(y_hat - y)
        # 同步更新参数
        w = w - alpha * dw
        b = b - alpha * db
        # 记录代价变化
        cost = compute_cost_logistic(X, y, w, b)
        cost_history.append(cost)
        # 每1000次迭代打印一次进度
        if i % 1000 == 0:
            print(f"迭代次数: {i:4d}, 代价: {cost:.4f}, w: {w:.4f}, b: {b:.4f}")

    return w, b, cost_history

# ==========================================
# 3. 训练模型
# ==========================================
# 初始化参数
w_init = 0.0
b_init = 0.0
# 学习率与迭代次数
alpha = 0.1
num_iters = 10000

print("开始训练逻辑回归模型...")
w_final, b_final, cost_history = gradient_descent_logistic(X, y, w_init, b_init, alpha, num_iters)
print(f"训练完成！最终参数: w = {w_final:.4f}, b = {b_final:.4f}")

# ==========================================
# 4. 可视化结果（中文正常显示）
# ==========================================
plt.figure(figsize=(12, 5))

# 图1：代价函数收敛曲线
plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.title("代价函数收敛曲线")
plt.xlabel("迭代次数")
plt.ylabel("代价 J(w,b)")

# 图2：数据点、Sigmoid曲线与决策边界
plt.subplot(1, 2, 2)
# 绘制训练数据
plt.scatter(X[y == 0], y[y == 0], color='blue', label='良性 (0)')
plt.scatter(X[y == 1], y[y == 1], color='red', label='恶性 (1)')
# 绘制Sigmoid拟合曲线
X_plot = np.linspace(0, 10, 100)
z_plot = w_final * X_plot + b_final
y_plot = sigmoid(z_plot)
plt.plot(X_plot, y_plot, color='green', label='Sigmoid拟合曲线')
# 绘制决策边界（概率=0.5的位置）
decision_boundary = -b_final / w_final
plt.axvline(x=decision_boundary, color='purple', linestyle='--', label=f'决策边界 (x={decision_boundary:.2f}cm)')
# 绘制0.5概率参考线
plt.axhline(y=0.5, color='gray', linestyle=':', label='概率=0.5')

plt.title("肿瘤大小 vs 恶性概率")
plt.xlabel("肿瘤大小 (cm)")
plt.ylabel("恶性概率")
plt.legend()
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()

# ==========================================
# 5. 模型预测演示
# ==========================================
tumor_size = 4.5  # 4.5cm
z = w_final * tumor_size + b_final
probability = sigmoid(z)
prediction = 1 if probability >= 0.5 else 0
print(f"\n预测结果：")
print(f"肿瘤大小为 {tumor_size} cm，")
print(f"恶性概率为: {probability*100:.1f}%，")
print(f"分类结果: {'恶性 (1)' if prediction == 1 else '良性 (0)'}")