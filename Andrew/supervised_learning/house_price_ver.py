import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===================== 【强制】中文乱码永久修复 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ====================================================================

# ==========================================
# 1. 准备数据（对应吴恩达课程房价预测）
# ==========================================
# 特征 X：房屋面积（单位：100平方英尺）
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
# 标签 y：房价（单位：1000美元）
y = np.array([1.2, 2.1, 2.8, 4.0, 5.1, 5.8, 7.2, 8.1])


# ==========================================
# 2. 定义核心函数
# ==========================================
def compute_cost(X, y, w, b):
    """
    均方误差成本函数
    对应吴恩达课程：J(w,b) = (1/(2m)) * sum( (h(x)-y)^2 )
    """
    m = X.shape[0]
    y_hat = w * X + b
    cost = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
    return cost


def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """
    批量梯度下降，同时记录w/b/cost的变化路径
    """
    m = X.shape[0]
    w, b = w_init, b_init
    w_history = [w]
    b_history = [b]
    cost_history = [compute_cost(X, y, w, b)]

    for i in range(num_iters):
        y_hat = w * X + b
        dw = (1 / m) * np.sum((y_hat - y) * X)
        db = (1 / m) * np.sum(y_hat - y)
        w -= alpha * dw
        b -= alpha * db
        w_history.append(w)
        b_history.append(b)
        cost_history.append(compute_cost(X, y, w, b))

    return w, b, w_history, b_history, cost_history


# ==========================================
# 3. 生成成本函数曲面数据
# ==========================================
print("正在生成成本函数曲面数据...")
# 生成w和b的取值网格
w_range = np.linspace(-0.5, 1.5, 100)  # w的范围：-0.5到1.5
b_range = np.linspace(-0.5, 1.5, 100)  # b的范围：-0.5到1.5
W, B = np.meshgrid(w_range, b_range)

# 计算每个(w,b)对应的成本J(w,b)
J = np.zeros_like(W)
for i in range(len(w_range)):
    for j in range(len(b_range)):
        J[j, i] = compute_cost(X, y, W[j, i], B[j, i])

# ==========================================
# 4. 运行梯度下降，获取优化路径
# ==========================================
print("正在运行梯度下降...")
w_init, b_init = 0.0, 0.0  # 初始参数
alpha = 0.01  # 学习率
num_iters = 1000  # 迭代次数
w_final, b_final, w_hist, b_hist, cost_hist = gradient_descent(
    X, y, w_init, b_init, alpha, num_iters
)
print(f"梯度下降完成！最优参数：w={w_final:.4f}, b={b_final:.4f}")

# ==========================================
# 5. 可视化成本函数（3D曲面 + 等高线）
# ==========================================
fig = plt.figure(figsize=(18, 7))

# --------------------------
# 图1：3D成本函数曲面 + 梯度下降路径
# --------------------------
ax1 = fig.add_subplot(121, projection='3d')
# 绘制碗状曲面
surf = ax1.plot_surface(W, B, J, cmap='viridis', alpha=0.7, edgecolor='none')
# 绘制梯度下降的路径（红色点线）
ax1.plot(w_hist, b_hist, cost_hist, color='red', linewidth=2, marker='o', markersize=3, label='梯度下降路径')
# 标记初始点和最优解
ax1.scatter(w_hist[0], b_hist[0], cost_hist[0], color='blue', s=100, marker='X', label='初始点 (w=0, b=0)')
ax1.scatter(w_final, b_final, cost_hist[-1], color='gold', s=150, marker='*', label='最优解 (碗底)')

ax1.set_xlabel('参数 w (斜率)', fontsize=12)
ax1.set_ylabel('参数 b (截距)', fontsize=12)
ax1.set_zlabel('成本 J(w,b)', fontsize=12)
ax1.set_title('3D成本函数曲面：梯度下降下山过程', fontsize=14)
ax1.legend(fontsize=10)
ax1.view_init(elev=30, azim=45)  # 调整视角，方便观察

# --------------------------
# 图2：成本函数等高线图 + 梯度下降路径
# --------------------------
ax2 = fig.add_subplot(122)
# 绘制等高线（颜色越深，成本越低）
contour = ax2.contour(W, B, J, levels=50, cmap='viridis')
plt.colorbar(contour, ax=ax2, label='成本 J(w,b)')
# 绘制梯度下降路径
ax2.plot(w_hist, b_hist, color='red', linewidth=2, marker='o', markersize=3, label='梯度下降路径')
# 标记初始点和最优解
ax2.scatter(w_hist[0], b_hist[0], color='blue', s=100, marker='X', label='初始点')
ax2.scatter(w_final, b_final, color='gold', s=150, marker='*', label='最优解')

ax2.set_xlabel('参数 w (斜率)', fontsize=12)
ax2.set_ylabel('参数 b (截距)', fontsize=12)
ax2.set_title('成本函数等高线图：梯度下降走向最优解', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 6. 结果解读
# ==========================================
print("\n=== 可视化结果解读 ===")
print("1. 左图3D曲面：碗状结构，底部是成本最低的最优解。")
print("2. 右图等高线：越靠近中心，颜色越深，成本越低。")
print("3. 红色路径：梯度下降从初始点(0,0)出发，一步步走向碗底。")
print("4. 这就是吴恩达课程讲的：梯度下降=在成本曲面上“下山”找最低点。")