import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork:
    """
    人工神经网络类，用于手写数字识别
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化神经网络
        参数:
            input_size: 输入层神经元数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出层神经元数量
        """
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        """
        ReLU激活函数
        """
        return np.maximum(0, x)
    
    def softmax(self, x):
        """
        Softmax激活函数
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        前向传播
        参数:
            X: 输入数据，形状为(n_samples, input_size)
        返回:
            预测结果，形状为(n_samples, output_size)
        """
        # 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        # 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        """
        反向传播
        参数:
            X: 输入数据
            y: 真实标签（独热编码）
            learning_rate: 学习率
        """
        m = X.shape[0]
        
        # 输出层梯度
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # 隐藏层梯度
        dZ1 = np.dot(dZ2, self.W2.T) * (self.a1 > 0)  # ReLU导数
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # 更新参数
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs, learning_rate):
        """
        训练模型
        参数:
            X: 训练数据
            y: 真实标签（独热编码）
            epochs: 训练轮数
            learning_rate: 学习率
        """
        for i in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            
            # 计算损失
            loss = -np.mean(np.sum(y * np.log(y_pred + 1e-10), axis=1))
            
            # 反向传播
            self.backward(X, y, learning_rate)
            
            # 每10轮显示一次损失
            if (i+1) % 10 == 0:
                print(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        预测
        参数:
            X: 输入数据
        返回:
            预测类别
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

def visualize_predictions(X, y_true, y_pred, num_samples=10):
    """
    可视化预测结果
    参数:
        X: 输入数据
        y_true: 真实标签
        y_pred: 预测标签
        num_samples: 显示的样本数量
    """
    plt.figure(figsize=(15, 4))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(X[i].reshape(8, 8), cmap='gray')
        plt.title(f"True: {y_true[i]}, Pred: {y_pred[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数，实现手写数字识别demo
    """
    # 加载数据
    print("正在加载手写数字数据集...")
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"数据集加载完成，共 {X.shape[0]} 个样本")
    
    # 数据预处理
    X = X / 16.0  # 归一化到[0,1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"数据分割完成，训练集: {X_train.shape[0]} 个样本, 测试集: {X_test.shape[0]} 个样本")
    
    # 标签独热编码
    encoder = OneHotEncoder()
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1)).toarray()
    
    # 创建并训练模型
    print("\n正在创建并训练神经网络...")
    nn = NeuralNetwork(input_size=64, hidden_size=64, output_size=10)
    nn.train(X_train, y_train_onehot, epochs=100, learning_rate=0.5)
    
    # 评估模型
    print("\n正在评估模型...")
    y_pred = nn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 可视化预测结果
    print("\n可视化预测结果...")
    visualize_predictions(X_test, y_test, y_pred)

if __name__ == "__main__":
    main()