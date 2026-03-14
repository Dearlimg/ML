import numpy as np

class NeuralNetwork:
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
            
            # 每次迭代都显示损失值
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

if __name__ == "__main__":
    # 示例：使用MNIST数据集训练简单的神经网络
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        import sys
        
        print(f"Python版本: {sys.version}")
        
        # 加载数据
        print("正在加载MNIST数据集...")
        # 使用不同的方式加载MNIST数据集，避免可能的网络问题
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data, digits.target
        print(f"数据集加载完成，X形状: {X.shape}, y形状: {y.shape}")
        y = y.astype(int)
        
        # 数据预处理
        X = X / 16.0  # 归一化到[0,1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"数据分割完成，训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
        
        # 标签独热编码
        print("正在进行独热编码...")
        encoder = OneHotEncoder()
        y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_test_onehot = encoder.transform(y_test.reshape(-1, 1)).toarray()
        print(f"独热编码完成，y_train_onehot形状: {y_train_onehot.shape}, y_test_onehot形状: {y_test_onehot.shape}")
        
        # 创建并训练模型
        print("正在创建并训练神经网络...")
        nn = NeuralNetwork(input_size=64, hidden_size=64, output_size=10)  # 增加隐藏层神经元数量
        nn.train(X_train, y_train_onehot, epochs=100, learning_rate=0.5)  # 增加训练轮数并提高学习率
        
        # 评估模型
        y_pred = nn.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"测试集准确率: {accuracy:.4f}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()