import numpy as np
import matplotlib.pyplot as plt

# ===================== 【强制】中文乱码永久修复 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ====================================================================

# ==========================================
# 1. 准备数据（对应吴恩达课程：训练集）
# ==========================================
# 特征 X：房屋面积（单位：100平方英尺）
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
# 标签 y：房价（单位：1000美元）
y = np.array([1.2, 2.1, 2.8, 4.0, 5.1, 5.8, 7.2, 8.1])

print("=== 训练数据 ===")
for i in range(len(X)):
    print(f"房屋面积: {X[i] * 100:.0f} 平方英尺 -> 房价: {y[i] * 1000:.0f} 美元")


# ==========================================
# 2. 定义核心函数（对应Mitchell教材：假设空间 + 性能度量）
# ==========================================
def hypothesis(x, w, b):
    """
    假设函数（Hypothesis）：对应Mitchell教材的「假设h」
    公式：h(x) = w * x + b
    """
    return w * x + b


def compute_cost(X, y, w, b):
    """
    代价函数（Cost Function）：对应吴恩达课程的「性能度量P」
    公式：J(w,b) = (1/(2m)) * sum( (h(x) - y)^2 )
    目标：最小化这个代价
    """
    m = X.shape[0]  # 训练样本数量
    y_hat = hypothesis(X, w, b)  # 模型的预测值
    cost = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
    return cost


def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """
    批量梯度下降（Batch Gradient Descent）：对应Mitchell教材的「搜索算法」
    目标：在假设空间里搜索，找到让代价最小的(w,b)
    """
    m = X.shape[0]
    w = w_init
    b = b_init
    cost_history = []  # 记录每次迭代的代价，用于观察收敛

    for i in range(num_iters):
        # 1. 计算预测值
        y_hat = hypothesis(X, w, b)

        # 2. 计算梯度（偏导数）
        # dw = dJ/dw, db = dJ/db
        dw = (1 / m) * np.sum((y_hat - y) * X)
        db = (1 / m) * np.sum(y_hat - y)

        # 3. 同时更新参数（吴恩达课程强调：必须同时更新！）
        w = w - alpha * dw
        b = b - alpha * db

        # 4. 记录当前代价
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        # 每100次迭代打印一次进度
        if i % 100 == 0:
            print(f"迭代次数: {i:4d} | 代价: {cost:.6f} | w: {w:.4f} | b: {b:.4f}")

    return w, b, cost_history


# ==========================================
# 3. 训练模型（对应Mitchell教材：通过经验E改进性能P）
# ==========================================
# 初始化参数（从0开始）
w_init = 0.0
b_init = 0.0
# 超参数设置（吴恩达课程核心调参点）
alpha = 0.01  # 学习率：不能太大（震荡），不能太小（收敛慢）
num_iters = 1000  # 迭代次数

print("\n=== 开始训练线性回归模型 ===")
w_final, b_final, cost_history = gradient_descent(X, y, w_init, b_init, alpha, num_iters)
print(f"=== 训练完成！ ===")
print(f"最终参数：w = {w_final:.4f}, b = {b_final:.4f}")
print(f"最终模型：房价 = {w_final:.2f} * 房屋面积 + {b_final:.2f}")

# ==========================================
# 4. 可视化结果（直观理解模型）
# ==========================================
plt.figure(figsize=(14, 6))

# 图1：代价函数收敛曲线
plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.title("代价函数收敛曲线")
plt.xlabel("迭代次数")
plt.ylabel("代价 J(w,b)")
plt.grid(True, alpha=0.3)

# 图2：训练数据与拟合直线
plt.subplot(1, 2, 2)
# 画出训练数据点
plt.scatter(X, y, color='blue', s=80, edgecolors='k', label='训练数据')
# 画出模型拟合的直线
X_line = np.linspace(0, 9, 100)
y_line = hypothesis(X_line, w_final, b_final)
plt.plot(X_line, y_line, color='red', linewidth=3, label=f'拟合直线: y = {w_final:.2f}x + {b_final:.2f}')
plt.title("房屋面积 vs 房价（线性回归拟合）")
plt.xlabel("房屋面积（100平方英尺）")
plt.ylabel("房价（1000美元）")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 5. 用训练好的模型做预测
# ==========================================
area = 6.5  # 550平方英尺
predicted_price = hypothesis(area, w_final, b_final)
print(f"\n=== 模型预测 ===")
print(f"输入：房屋面积 = {area * 100:.0f} 平方英尺")
print(f"预测：房价约为 {predicted_price * 1000:.0f} 美元")