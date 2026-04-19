import numpy as np
import matplotlib.pyplot as plt

# ===================== 实验3：梯度下降算法实现 =====================

print("=" * 50)
print("实验3：梯度下降算法实现")
print("=" * 50)

# 1. 定义目标函数 f(x) = x^2
def objective_function(x):
    return x ** 2

# 2. 定义目标函数的梯度（导数）f'(x) = 2x
def gradient(x):
    return 2 * x

# 3. 梯度下降算法
def gradient_descent(learning_rate, epochs, initial_x=5.0):
    x = initial_x
    history = [x]
    print(f"\n初始点: x = {x}, f(x) = {objective_function(x):.4f}")

    for epoch in range(epochs):
        grad = gradient(x)
        x = x - learning_rate * grad
        history.append(x)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: x = {x:10.6f}, f(x) = {objective_function(x):10.6f}, gradient = {grad:.6f}")

    return x, history

# 4. 运行不同学习率的实验
print("\n" + "=" * 50)
print("不同学习率的梯度下降实验")
print("=" * 50)

results = {}
for lr in [0.1, 0.5, 0.8, 1.0]:
    print(f"\n----- 学习率 lr = {lr} -----")
    final_x, history = gradient_descent(lr, epochs=20, initial_x=5.0)
    results[lr] = (final_x, history)

# 5. 可视化
plt.figure(figsize=(12, 5))

# 子图1：函数曲线和梯度下降路径
plt.subplot(1, 2, 1)
x_range = np.linspace(-6, 6, 100)
y_range = objective_function(x_range)
plt.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = x²')

colors = ['red', 'green', 'orange', 'purple']
for i, (lr, (final_x, history)) in enumerate(results.items()):
    history = np.array(history)
    plt.plot(history, objective_function(history), 'o-', color=colors[i],
             markersize=4, linewidth=1, label=f'lr={lr}')

plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Gradient Descent on f(x) = x²', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：收敛曲线
plt.subplot(1, 2, 2)
for i, (lr, (final_x, history)) in enumerate(results.items()):
    plt.plot(range(len(history)), history, 'o-', color=colors[i],
             markersize=4, linewidth=1.5, label=f'lr={lr}')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('x', fontsize=12)
plt.title('Convergence of x', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_descent.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\n图形已保存: gradient_descent.png")

# 6. 不同初始点的实验
print("\n" + "=" * 50)
print("不同初始点的梯度下降实验 (lr=0.1)")
print("=" * 50)

for initial in [5.0, -5.0, 3.0, -3.0]:
    print(f"\n初始点: x = {initial}")
    final_x, history = gradient_descent(0.1, epochs=20, initial_x=initial)
    print(f"最终结果: x = {final_x:.6f}, f(x) = {objective_function(final_x):.6f}")

print("\n" + "=" * 50)
print("实验3完成")
print("=" * 50)