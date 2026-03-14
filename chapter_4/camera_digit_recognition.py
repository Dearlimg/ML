import cv2
import numpy as np
from sklearn.datasets import load_digits
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

def preprocess_image(img):
    """
    预处理图像
    参数:
        img: 输入图像
    返回:
        预处理后的图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊，减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值二值化，适应不同光照条件
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形态学操作，去除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def find_contours(thresh):
    """
    找到图像中的轮廓
    参数:
        thresh: 二值化图像
    返回:
        轮廓列表
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def process_contour(contour, img):
    """
    处理轮廓，提取数字区域
    参数:
        contour: 轮廓
        img: 原始图像
    返回:
        处理后的数字图像，边界框
    """
    # 计算轮廓面积
    area = cv2.contourArea(contour)
    
    # 计算边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # 过滤太小的轮廓
    if w < 30 or h < 30 or area < 200:
        return None, None
    
    # 检查宽高比，数字通常接近正方形，但3可能更宽
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.4 or aspect_ratio > 1.8:
        return None, None
    
    # 计算轮廓周长
    perimeter = cv2.arcLength(contour, True)
    
    # 检查轮廓的紧凑度，3的紧凑度可能较低
    if perimeter == 0:
        return None, None
    compactness = 4 * np.pi * area / (perimeter * perimeter)
    if compactness < 0.1:
        return None, None
    
    # 扩展边界框，确保包含整个数字
    x = max(0, x - 15)
    y = max(0, y - 15)
    w = min(img.shape[1] - x, w + 30)
    h = min(img.shape[0] - y, h + 30)
    
    # 提取数字区域
    roi = img[y:y+h, x:x+w]
    
    # 转换为灰度图
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值二值化
    thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 调整大小为8x8，与训练数据一致
    resized = cv2.resize(thresh_roi, (8, 8), interpolation=cv2.INTER_AREA)
    
    # 归一化到[0,1]
    normalized = resized / 255.0
    
    # 展平为64维向量
    flattened = normalized.flatten().reshape(1, -1)
    
    return flattened, (x, y, w, h)

def train_model():
    """
    训练神经网络模型
    返回:
        训练好的模型
    """
    # 加载数据
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 数据预处理
    X = X / 16.0  # 归一化到[0,1]
    
    # 标签独热编码
    encoder = OneHotEncoder()
    y_onehot = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    
    # 创建并训练模型
    nn = NeuralNetwork(input_size=64, hidden_size=128, output_size=10)  # 增加隐藏层神经元数量
    nn.train(X, y_onehot, epochs=200, learning_rate=0.3)  # 增加训练轮数，调整学习率
    
    return nn

def main():
    """
    主函数，实现摄像头手写数字识别
    """
    # 训练模型
    print("正在训练模型...")
    model = train_model()
    print("模型训练完成！")
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("摄像头启动成功！请在摄像头前写数字，按 'q' 退出")
    
    while True:
        # 捕获帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 水平翻转，使镜像效果更自然
        frame = cv2.flip(frame, 1)
        
        # 预处理图像
        thresh = preprocess_image(frame)
        
        # 找到轮廓
        contours = find_contours(thresh)
        
        # 处理每个轮廓
        for contour in contours:
            # 提取数字区域
            digit_img, bbox = process_contour(contour, frame)
            if digit_img is not None:
                # 预测数字
                prediction = model.predict(digit_img)[0]
                
                # 绘制边界框和预测结果
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, str(prediction), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Handwritten Digit Recognition', frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()