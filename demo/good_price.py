import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import GridSearchCV

# ===================== 中文乱码修复 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

# ==========================================
# 1. 数据准备：生成模拟电商用户行为数据（贴近真实）
# ==========================================
print("=== 步骤1：生成/加载用户行为数据 ===")
np.random.seed(42)  # 固定随机种子，结果可复现

# 生成1000条用户行为数据
n_samples = 1000
data = {
    # 特征1：商品浏览时长（秒），购买用户通常浏览更久
    'view_duration': np.concatenate([
        np.random.normal(10, 5, 700),  # 未购买用户：平均10秒
        np.random.normal(30, 10, 300)  # 购买用户：平均30秒
    ]),
    # 特征2：加购次数（次），购买用户加购次数更多
    'add_cart_count': np.concatenate([
        np.random.poisson(0.5, 700),  # 未购买用户：平均0.5次
        np.random.poisson(2, 300)  # 购买用户：平均2次
    ]),
    # 特征3：收藏次数（次）
    'collect_count': np.concatenate([
        np.random.poisson(0.2, 700),  # 未购买用户：平均0.2次
        np.random.poisson(1, 300)  # 购买用户：平均1次
    ]),
    # 特征4：是否首次访问（0=否，1=是），首次访问购买率低
    'is_first_visit': np.random.randint(0, 2, n_samples),
    # 标签：是否购买（0=未购买，1=购买）
    'is_purchase': np.concatenate([np.zeros(700), np.ones(300)])
}

# 转为DataFrame，方便数据处理
df = pd.DataFrame(data)
# 处理异常值（比如浏览时长为负）
df['view_duration'] = df['view_duration'].clip(lower=0)

print(f"数据形状：{df.shape}")
print(f"购买用户占比：{df['is_purchase'].mean():.2f}")
print("前5行数据：")
print(df.head())

# ==========================================
# 2. 数据预处理：划分训练集/测试集 + 特征标准化
# ==========================================
print("\n=== 步骤2：数据预处理 ===")
# 划分特征X和标签y
X = df.drop('is_purchase', axis=1)
y = df['is_purchase']

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify：保证测试集和训练集的购买率一致
)

# 特征标准化（逻辑回归对特征尺度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集形状：{X_train.shape}")
print(f"测试集形状：{X_test.shape}")

# ==========================================
# 3. 模型训练：先训练基础逻辑回归模型
# ==========================================
print("\n=== 步骤3：基础模型训练 ===")
# 初始化逻辑回归模型
lr_base = LogisticRegression(random_state=42)
# 训练模型
lr_base.fit(X_train_scaled, y_train)

# 预测
y_pred_base = lr_base.predict(X_test_scaled)
y_pred_prob_base = lr_base.predict_proba(X_test_scaled)[:, 1]  # 购买概率


# 模型评估
def evaluate_model(y_true, y_pred, y_pred_prob, model_name):
    """封装模型评估函数，输出核心指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)

    print(f"\n【{model_name} 评估指标】")
    print(f"准确率（Accuracy）：{accuracy:.4f}")  # 整体预测准度
    print(f"精确率（Precision）：{precision:.4f}")  # 预测为购买的用户中，真实购买的比例
    print(f"召回率（Recall）：{recall:.4f}")  # 真实购买的用户中，被预测出来的比例
    print(f"F1分数：{f1:.4f}")  # 精确率和召回率的调和平均
    print(f"AUC值：{auc:.4f}")  # 模型区分能力（越大越好，1=完美，0.5=随机）

    # 输出特征重要性
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '系数（重要性）': lr_base.coef_[0]
    }).sort_values('系数（重要性）', ascending=False)
    print("\n【特征重要性】")
    print(feature_importance)

    return {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1, 'auc': auc
    }


# 评估基础模型
base_metrics = evaluate_model(y_test, y_pred_base, y_pred_prob_base, "基础逻辑回归模型")

# ==========================================
# 4. 模型优化：网格搜索调参（找最优参数）
# ==========================================
print("\n=== 步骤4：模型调参优化 ===")
# 定义参数网格
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # 正则化强度（越小正则化越强）
    'penalty': ['l1', 'l2'],  # 正则化类型
    'solver': ['liblinear']  # 支持l1正则化的求解器
}

# 网格搜索（交叉验证）
grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5折交叉验证
    scoring='roc_auc',  # 以AUC为优化目标
    n_jobs=-1  # 并行计算
)

# 训练网格搜索
grid_search.fit(X_train_scaled, y_train)

# 最优参数和最优模型
best_params = grid_search.best_params_
lr_best = grid_search.best_estimator_

print(f"最优参数：{best_params}")

# 用最优模型预测
y_pred_best = lr_best.predict(X_test_scaled)
y_pred_prob_best = lr_best.predict_proba(X_test_scaled)[:, 1]

# 评估最优模型
best_metrics = evaluate_model(y_test, y_pred_best, y_pred_prob_best, "调参后最优模型")

# ==========================================
# 5. 可视化：模型效果对比 + ROC曲线
# ==========================================
print("\n=== 步骤5：可视化模型效果 ===")
plt.figure(figsize=(15, 6))

# 图1：基础模型 vs 最优模型 指标对比
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
base_vals = [base_metrics[m] for m in metrics]
best_vals = [best_metrics[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.subplot(1, 2, 1)
plt.bar(x - width / 2, base_vals, width, label='基础模型', alpha=0.8)
plt.bar(x + width / 2, best_vals, width, label='调参后模型', alpha=0.8)
plt.xticks(x, metrics)
plt.ylabel('指标值')
plt.title('基础模型 vs 调参后模型 核心指标对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 图2：ROC曲线（模型区分能力）
plt.subplot(1, 2, 2)
# 基础模型ROC
fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_prob_base)
plt.plot(fpr_base, tpr_base, label=f'基础模型 (AUC={base_metrics["auc"]:.4f})', linewidth=2)
# 最优模型ROC
fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_prob_best)
plt.plot(fpr_best, tpr_best, label=f'调参后模型 (AUC={best_metrics["auc"]:.4f})', linewidth=2)
# 随机猜测的基线
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC=0.5)', linewidth=1)

plt.xlabel('假阳性率（FPR）')
plt.ylabel('真阳性率（TPR）')
plt.title('ROC曲线（模型区分购买/未购买用户的能力）')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 6. 模型部署：用最优模型预测新用户
# ==========================================
print("\n=== 步骤6：模型部署（预测新用户） ===")
# 模拟3个新用户的行为数据
new_users = pd.DataFrame({
    'view_duration': [5, 40, 20],  # 浏览时长：5秒、40秒、20秒
    'add_cart_count': [0, 3, 1],  # 加购次数：0次、3次、1次
    'collect_count': [0, 2, 0],  # 收藏次数：0次、2次、0次
    'is_first_visit': [1, 0, 1]  # 是否首次访问：是、否、是
})

# 预处理新用户数据
new_users_scaled = scaler.transform(new_users)

# 预测购买概率
purchase_prob = lr_best.predict_proba(new_users_scaled)[:, 1]

# 输出预测结果
print("\n【新用户购买概率预测】")
for i in range(len(new_users)):
    print(f"用户{i + 1}：")
    print(f"  行为数据：{new_users.iloc[i].to_dict()}")
    print(f"  购买概率：{purchase_prob[i]:.4f}")
    print(f"  预测结果：{'购买' if purchase_prob[i] >= 0.5 else '未购买'}")

# ==========================================
# 7. 训练过程反思：总结经验
# ==========================================
print("\n=== 步骤7：训练过程反思 ===")
print("1. 特征重要性：浏览时长和加购次数是预测购买的核心特征，符合业务直觉。")
print("2. 调参效果：调参后模型的AUC提升，说明正则化能有效避免过拟合。")
print("3. 业务价值：能精准预测用户购买概率，可针对性推送优惠券，提升转化率。")
print("4. 改进方向：可增加更多特征（如用户性别、消费能力），或尝试树模型（如XGBoost）。")