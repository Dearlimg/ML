import cv2
import numpy as np

# ===================== 中文乱码修复（针对Windows） =====================
# 注：opencv的窗口标题中文可能仍显示异常，但不影响核心功能
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ========================================================================

class GenderDetectionDemo:
    def __init__(self):
        # 1. 加载人脸检测模型（opencv自带的 Haar级联分类器，轻量快速）
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # 2. 加载预训练的性别分类模型（基于CNN，轻量高效）
        # 模型说明：输入为64x64灰度图，输出为[男性概率, 女性概率]
        self.gender_model = cv2.dnn.readNetFromCaffe(
            # 模型配置文件（定义网络结构）
            prototxt="gender_deploy.prototxt",
            # 模型权重文件（预训练参数）
            caffeModel="gender_net.caffemodel"
        )

        # 性别标签
        self.gender_labels = ['Male', 'Female']
        # 摄像头对象（0=默认前置摄像头，部分设备需改为1）
        self.cap = cv2.VideoCapture(0)

        # 检查摄像头是否打开
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头！请检查摄像头是否被占用，或尝试修改摄像头索引（0→1）。")

    def preprocess_face(self, face_img):
        """预处理人脸图像：转为64x64灰度图，符合模型输入要求"""
        # 转为灰度图
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # 调整尺寸为64x64
        resized_face = cv2.resize(gray_face, (64, 64))
        # 转为模型需要的格式（归一化、添加维度）
        blob = cv2.dnn.blobFromImage(
            resized_face,
            scalefactor=1.0 / 255,  # 归一化到0-1
            size=(64, 64),
            mean=0,  # 均值归一化
            swapRB=False  # 灰度图无需交换通道
        )
        return blob

    def predict_gender(self, face_img):
        """预测单个人脸的性别"""
        # 预处理人脸
        blob = self.preprocess_face(face_img)
        # 输入模型预测
        self.gender_model.setInput(blob)
        predictions = self.gender_model.forward()
        # 获取预测结果（概率最大的类别）
        gender_idx = np.argmax(predictions[0])
        gender = self.gender_labels[gender_idx]
        confidence = predictions[0][gender_idx] * 100  # 概率转百分比
        return gender, confidence

    def run(self):
        """启动实时检测"""
        print("=== 性别检测Demo启动 ===")
        print("提示：")
        print("1. 按 'q' 键退出程序")
        print("2. 按 's' 键保存当前帧到本地")
        print("3. 若摄像头无画面，尝试修改代码中摄像头索引（0→1）")

        while True:
            # 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面，退出...")
                break

            # 镜像翻转（前置摄像头画面通常是反的，更符合直觉）
            frame = cv2.flip(frame, 1)

            # 转为灰度图（人脸检测需要）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # 缩放因子，检测不同大小的人脸
                minNeighbors=5,  # 过滤误检
                minSize=(30, 30)  # 最小人脸尺寸
            )

            # 遍历检测到的人脸，逐个预测性别
            for (x, y, w, h) in faces:
                # 截取人脸区域
                face_img = frame[y:y + h, x:x + w]

                # 预测性别
                gender, confidence = self.predict_gender(face_img)

                # 绘制人脸框和预测结果
                # 框的颜色：男性(蓝色)，女性(粉色)
                color = (255, 0, 0) if gender == 'Male' else (255, 192, 203)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # 绘制文本背景（避免文字和画面重叠看不清）
                text = f"{gender} ({confidence:.1f}%)"
                cv2.rectangle(frame, (x, y - 30), (x + len(text) * 10, y), color, -1)
                cv2.putText(
                    frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

            # 显示画面
            cv2.imshow("Gender Detection (按q退出)", frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按q退出
                break
            elif key == ord('s'):  # 按s保存帧
                cv2.imwrite("gender_detection_screenshot.jpg", frame)
                print("已保存当前帧到：gender_detection_screenshot.jpg")

        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("=== 性别检测Demo退出 ===")


# ===================== 模型文件下载说明 =====================
def download_model_files():
    """提示用户下载必要的模型文件"""
    print("=== 模型文件下载提示 ===")
    print("请先下载以下两个文件，放在和代码同一目录下：")
    print("1. gender_deploy.prototxt（模型配置）：")
    print("   下载地址：https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_deploy.prototxt")
    print("2. gender_net.caffemodel（模型权重）：")
    print("   下载地址：https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/gender_net.caffemodel")
    print("\n下载方式：")
    print("1. 复制链接到浏览器下载，保存到代码文件夹；")
    print("2. 若无法访问GitHub，可私信我获取文件。")


if __name__ == "__main__":
    # 先提示下载模型文件
    download_model_files()

    # 等待用户确认
    input("\n请确认模型文件已下载并放在代码目录下，按Enter键启动...")

    # 启动Demo
    try:
        demo = GenderDetectionDemo()
        demo.run()
    except Exception as e:
        print(f"程序出错：{e}")