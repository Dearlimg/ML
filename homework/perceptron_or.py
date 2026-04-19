import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据
# 输入矩阵X：前两列是x1, x2，第三列是偏置项1
X = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

# 标签y：OR运算的结果（0或1）
y = np.array([0, 1, 1, 1])

# 2. 定义激活函数（阶跃函数）
def step_function(z):
    return 1 if z >= 0 else 0

# 3. 感知机训练函数
def perceptron_train(X, y, lr=0.1, epochs=10):
    # 初始化权重（包括偏置项）
    weights = np.zeros(X.shape[1])
    print("初始权重：", weights)

    for epoch in range(epochs):
        print(f"\n===== 第 {epoch+1} 轮训练 =====")
        for i in range(len(X)):
            # 计算加权和
            z = np.dot(X[i], weights)
            # 预测输出
            y_pred = step_function(z)
            # 更新权重：w = w + lr*(y_true - y_pred)*x
            error = y[i] - y_pred
            weights += lr * error * X[i]
            print(f"样本{i+1}：预测={y_pred}，实际={y[i]}，误差={error}，权重更新为：{weights}")
    return weights

# 4. 训练模型
weights = perceptron_train(X, y, lr=0.5, epochs=5)
print("\n训练完成，最终权重：", weights)

# 5. 测试模型
print("\n===== OR运算测试 =====")
correct = 0
for i in range(len(X)):
    z = np.dot(X[i], weights)
    y_pred = step_function(z)
    print(f"输入({X[i][0]}, {X[i][1]}) → 预测：{y_pred}，实际：{y[i]}")
    if y_pred == y[i]:
        correct += 1
print(f"测试准确率：{correct / len(X) * 100}%")

# 6. 封装预测函数
def or_predict(x1, x2, weights):
    # 加上偏置项1
    x = np.array([x1, x2, 1])
    z = np.dot(x, weights)
    return step_function(z)

# 自定义测试
print("\n===== 自定义测试 =====")
print(f"OR(0, 0) = {or_predict(0, 0, weights)}")
print(f"OR(0, 1) = {or_predict(0, 1, weights)}")
print(f"OR(1, 0) = {or_predict(1, 0, weights)}")
print(f"OR(1, 1) = {or_predict(1, 1, weights)}")

# 7. 图形化展示
print("\n===== 图形化展示 =====")

# 创建画布
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：数据点分布
ax1 = axes[0]
colors = ['red' if label == 0 else 'green' for label in y]
scatter = ax1.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidths=2, marker='o')

for i, (x1, x2) in enumerate(X[:, :2]):
    ax1.annotate(f'({int(x1)},{int(x2)})\ny={y[i]}', 
                xy=(x1, x2), xytext=(x1+0.15, x2+0.15),
                fontsize=11, fontweight='bold')

ax1.set_xlabel('x1', fontsize=12)
ax1.set_ylabel('x2', fontsize=12)
ax1.set_title('OR Logic Data Points', fontsize=14, fontweight='bold')
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Class 0'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, label='Class 1')],
           loc='upper left')
ax1.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5)
ax1.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5)

# 子图2：决策边界
ax2 = axes[1]

# 绘制决策区域
x1_range = np.linspace(-0.5, 1.5, 100)
x2_range = np.linspace(-0.5, 1.5, 100)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
z_grid = weights[0] * xx1 + weights[1] * xx2 + weights[2]
y_grid = np.where(z_grid >= 0, 1, 0)
ax2.contourf(xx1, xx2, y_grid, alpha=0.3, cmap=plt.cm.RdYlGn)
ax2.contour(xx1, xx2, y_grid, colors='blue', linewidths=2, levels=[0.5])

# 绘制数据点
scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidths=2, marker='o')
for i, (x1, x2) in enumerate(X[:, :2]):
    ax2.annotate(f'({int(x1)},{int(x2)})\ny={y[i]}',
                xy=(x1, x2), xytext=(x1+0.15, x2+0.15),
                fontsize=11, fontweight='bold')

# 绘制决策边界线
x1_line = np.array([-0.5, 1.5])
x2_line = -(weights[0] * x1_line + weights[2]) / weights[1]
ax2.plot(x1_line, x2_line, 'b-', linewidth=2, label=f'Decision Boundary\n{weights[0]:.2f}x1 + {weights[1]:.2f}x2 + {weights[2]:.2f} = 0')

ax2.set_xlabel('x1', fontsize=12)
ax2.set_ylabel('x2', fontsize=12)
ax2.set_title('OR Logic - Perceptron Decision Boundary', fontsize=14, fontweight='bold')
ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig("or_logic_perceptron.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("图形已保存: or_logic_perceptron.png")