
import numpy as np

# -------------------------
# 1 生成数据
# -------------------------

np.random.seed(0)

x = np.random.rand(100)
y = 3 * x + 2 + np.random.randn(100) * 0.1   # y = 3x + 2 + noise


# -------------------------
# 2 初始化参数
# -------------------------

w = 0
b = 0

learning_rate = 0.1
epochs = 1000
n = len(x)


# -------------------------
# 3 梯度下降
# -------------------------

for epoch in range(epochs):

    # 预测
    y_pred = w * x + b

    # loss
    loss = np.mean((y_pred - y) ** 2)

    # 计算梯度
    dw = (2/n) * np.sum(x * (y_pred - y))
    db = (2/n) * np.sum(y_pred - y)

    # 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # 打印训练过程
    if epoch % 100 == 0:
        print(f"epoch={epoch}, loss={loss:.4f}, w={w:.4f}, b={b:.4f}")


print("\n最终参数：")
print("w =", w)
print("b =", b)