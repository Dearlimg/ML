import numpy as np
import matplotlib.pyplot as plt

# ===================== 中文乱码修复核心代码 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# ==============================================================

# ==========================================
# 1. 准备数据（对应吴恩达课程房价数据集）
# ==========================================
# 特征：房屋面积（单位：100平方英尺）
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# 标签：房价（单位：1000美元）
y = np.array([1.0, 2.5, 3.0, 4.5, 5.0])

# ==========================================
# 2. 定义核心函数（对应吴恩达课程公式）
# ==========================================
def compute_cost(X, y, w, b):
    """
    计算均方误差代价函数
    对应吴恩达课程：J(w,b) = (1/(2m)) * sum((y_hat - y)^2)
    """
    m = X.shape[0]
    y_hat = w * X + b  # 线性模型的预测值
    cost = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
    return cost

def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """
    批量梯度下降算法
    对应吴恩达课程：同时更新w和b
    """
    m = X.shape[0]
    w = w_init
    b = b_init
    cost_history = []  # 记录每次迭代的代价，用于观察收敛

    for i in range(num_iters):
        # 计算预测值
        y_hat = w * X + b
        # 计算梯度（偏导数）
        dw = (1 / m) * np.sum((y_hat - y) * X)
        db = (1 / m) * np.sum(y_hat - y)
        # 同时更新参数
        w = w - alpha * dw
        b = b - alpha * db
        # 记录代价
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        # 每100次迭代打印一次
        if i % 100 == 0:
            print(f"迭代次数: {i:4d}, 代价: {cost:.4f}, w: {w:.4f}, b: {b:.4f}")

    return w, b, cost_history

# ==========================================
# 3. 训练模型
# ==========================================
# 初始化参数
w_init = 0.0
b_init = 0.0
# 学习率（吴恩达课程核心超参数）
alpha = 0.01
# 迭代次数
num_iters = 1000

print("开始训练线性回归模型...")
w_final, b_final, cost_history = gradient_descent(X, y, w_init, b_init, alpha, num_iters)
print(f"训练完成！最终参数: w = {w_final:.4f}, b = {b_final:.4f}")

# ==========================================
# 4. 可视化结果（中文已修复）
# ==========================================
plt.figure(figsize=(12, 5))

# 图1：代价函数收敛曲线
plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.title("代价函数收敛曲线")
plt.xlabel("迭代次数")
plt.ylabel("代价 J(w,b)")

# 图2：数据点与拟合直线
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='训练数据')
plt.plot(X, w_final * X + b_final, color='red', label=f'拟合直线: y = {w_final:.2f}x + {b_final:.2f}')
plt.title("房屋面积 vs 房价")
plt.xlabel("房屋面积 (100平方英尺)")
plt.ylabel("房价 (1000美元)")
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 5. 模型预测
# ==========================================
area = 3.5  # 350平方英尺
predicted_price = w_final * area + b_final
print(f"\n预测：面积为 {area*100:.0f} 平方英尺的房屋，房价约为 {predicted_price*1000:.0f} 美元")