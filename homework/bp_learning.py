import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

print("=" * 50)
print("实验4：BP学习算法的实现")
print("=" * 50)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        delta2 = (y - self.a2) * sigmoid_derivative(self.a2)
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        self.W1 += self.learning_rate * dW1
        self.b1 += self.learning_rate * db1
        self.W2 += self.learning_rate * dW2
        self.b2 += self.learning_rate * db2

    def train(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32):
        self.loss_history = []
        self.val_loss_history = []
        n_samples = X.shape[0]
        print("\n===== BP训练过程 =====")
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            self.loss_history.append(loss)
            if X_val is not None:
                val_output = self.forward(X_val)
                val_loss = np.mean((y_val - val_output) ** 2)
                self.val_loss_history.append(val_loss)
            if (epoch + 1) % 10 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch+1:3d}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1:3d}: Loss = {loss:.4f}")
        predictions = self.predict(X)
        accuracy = np.mean(predictions == np.argmax(y, axis=1)) * 100
        print(f"\n最终训练集准确率: {accuracy:.2f}%")
        print("训练完成！")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

print("\n===== 加载手写数字数据集 =====")
digits = load_digits()
X = digits.data / 16.0
y_onehot = np.zeros((len(digits.target), 10))
for i, target in enumerate(digits.target):
    y_onehot[i, target] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"输入特征维度: {X_train.shape[1]}")
print(f"输出类别数: 10 (数字0-9)")

print("\n===== BP网络训练 =====")
print("网络结构: 64 -> 32 -> 10")
print("激活函数: Sigmoid")
print("学习率: 0.5")

bpnn = BPNeuralNetwork(input_size=64, hidden_size=32, output_size=10, learning_rate=0.5)
bpnn.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=64)

print("\n===== 测试集预测结果 =====")
predictions = bpnn.predict(X_test)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == true_labels) * 100
print(f"测试集准确率: {accuracy:.2f}%")

print("\n部分预测示例:")
print("真实标签 | 预测结果 | 正确")
print("---------|----------|------")
for i in range(min(10, len(predictions))):
    correct = "√" if predictions[i] == true_labels[i] else "×"
    print(f"   {true_labels[i]}      |    {predictions[i]}     |  {correct}")

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(bpnn.loss_history, 'b-', linewidth=1.5, label='Train Loss')
if bpnn.val_loss_history:
    ax1.plot(bpnn.val_loss_history, 'r-', linewidth=1.5, label='Val Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('BP Training Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title('Sample Predictions', fontsize=14, fontweight='bold')
ax2.axis('off')
sample_indices = np.random.choice(len(X_test), 16, replace=False)
for i, idx in enumerate(sample_indices):
    subax = fig.add_subplot(4, 4, i + 1)
    subax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    subax.axis('off')
    color = 'green' if predictions[idx] == true_labels[idx] else 'red'
    subax.set_title(f'{predictions[idx]}', fontsize=10, color=color)

ax3 = fig.add_subplot(1, 3, 3)
ax3.axis('off')
layer_sizes = [64, 32, 10]
layer_names = ['Input\n(64)', 'Hidden\n(32)', 'Output\n(10)']
max_display = 8

for layer_idx, (n_neurons, name) in enumerate(zip(layer_sizes, layer_names)):
    display_neurons = min(n_neurons, max_display)
    y_positions = np.linspace(0.15, 0.85, display_neurons)
    for y_pos in y_positions:
        circle = plt.Circle((layer_idx * 0.4 + 0.2, y_pos), 0.04, color='lightblue', ec='blue', linewidth=2)
        ax3.add_patch(circle)
    if n_neurons > max_display:
        ax3.text(layer_idx * 0.4 + 0.2, 0.08, f'...+{n_neurons - max_display}', ha='center', va='center', fontsize=9)
    ax3.text(layer_idx * 0.4 + 0.2, 0.95, name, ha='center', va='center', fontsize=11, fontweight='bold')
    if layer_idx < len(layer_sizes) - 1:
        next_n_neurons = layer_sizes[layer_idx + 1]
        next_display = min(next_n_neurons, max_display)
        next_y_positions = np.linspace(0.15, 0.85, next_display)
        for y1 in y_positions:
            for y2 in next_y_positions:
                ax3.plot([layer_idx * 0.4 + 0.24, (layer_idx + 1) * 0.4 + 0.16], [y1, y2], 'b-', alpha=0.1, linewidth=0.3)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title('BP Network Structure', fontsize=14, fontweight='bold')

plt.savefig("bp_neural_network.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\n图形已保存: bp_neural_network.png")

print("\n" + "=" * 50)
print("BP学习算法核心步骤:")
print("1. 前向传播: 输入 -> 隐藏层 -> 输出层")
print("2. 计算损失: MSE = (y - output)^2")
print("3. 反向传播: 输出层误差 -> 隐藏层误差 -> 更新权重")
print("4. 迭代优化: 重复步骤1-3直到收敛")
print("=" * 50)
print("实验4完成")
print("=" * 50)