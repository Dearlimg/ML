import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 1. 加载数据集 =====================
def create_data():
    # 对应题目中的34条数据
    data = [
        ['坏', '是', '是', '高'], ['坏', '是', '是', '高'], ['坏', '是', '是', '高'],
        ['坏', '否', '是', '高'], ['坏', '是', '是', '高'], ['坏', '否', '是', '高'],
        ['坏', '是', '否', '高'], ['好', '是', '是', '高'], ['好', '是', '否', '高'],
        ['好', '是', '是', '高'], ['好', '是', '是', '高'], ['好', '是', '是', '高'],
        ['好', '是', '是', '高'], ['坏', '是', '是', '低'], ['好', '否', '是', '高'],
        ['好', '否', '是', '高'], ['好', '否', '是', '高'], ['好', '否', '是', '高'],
        ['好', '否', '否', '高'], ['坏', '否', '否', '低'], ['坏', '否', '是', '低'],
        ['坏', '否', '是', '低'], ['坏', '否', '是', '低'], ['坏', '否', '否', '低'],
        ['坏', '是', '否', '低'], ['好', '否', '是', '低'], ['好', '否', '是', '低'],
        ['坏', '否', '否', '低'], ['坏', '否', '否', '低'], ['好', '否', '否', '低'],
        ['坏', '是', '否', '低'], ['好', '否', '是', '低'], ['好', '否', '否', '低'],
        ['好', '否', '否', '低']
    ]
    # 特征名称：天气、是否周末、是否有促销；标签：销量
    labels = ['天气', '是否周末', '是否有促销']
    return data, labels


# ===================== 2. 计算信息熵 =====================
def calc_entropy(data):
    """
    计算数据集的信息熵
    :param data: 数据集
    :return: 信息熵
    """
    sample_num = len(data)
    # 统计每个类别（销量）的数量
    label_counts = {}
    for sample in data:
        label = sample[-1]
        label_counts[label] = label_counts.get(label, 0) + 1

    entropy = 0.0
    for count in label_counts.values():
        p = count / sample_num
        entropy -= p * math.log2(p)
    return entropy


# ===================== 3. 按特征值划分数据集 =====================
def split_data(data, axis, value):
    """
    根据特征划分数据集
    :param data: 原始数据集
    :param axis: 特征索引
    :param value: 特征值
    :return: 划分后的数据集
    """
    new_data = []
    for sample in data:
        if sample[axis] == value:
            # 去掉当前特征，保留剩余特征和标签
            reduced_sample = sample[:axis] + sample[axis + 1:]
            new_data.append(reduced_sample)
    return new_data


# ===================== 4. 选择最优特征（信息增益最大） =====================
def choose_best_feature(data):
    """
    选择信息增益最大的特征
    :return: 最优特征索引
    """
    feature_num = len(data[0]) - 1  # 特征数量（去掉标签列）
    base_entropy = calc_entropy(data)  # 原始信息熵
    best_gain = 0.0
    best_feature_idx = -1

    for i in range(feature_num):
        # 获取当前特征的所有取值
        feature_values = [sample[i] for sample in data]
        unique_values = set(feature_values)
        new_entropy = 0.0

        # 计算条件熵
        for value in unique_values:
            sub_data = split_data(data, i, value)
            p = len(sub_data) / len(data)
            new_entropy += p * calc_entropy(sub_data)

        # 计算信息增益
        info_gain = base_entropy - new_entropy
        # 更新最优特征
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature_idx = i
    return best_feature_idx


# ===================== 5. 构建决策树 =====================
def create_tree(data, labels):
    """
    递归构建决策树
    :return: 决策树字典
    """
    # 获取所有标签
    label_list = [sample[-1] for sample in data]
    # 终止条件1：所有样本属于同一类别
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # 终止条件2：没有特征可划分
    if len(data[0]) == 1:
        # 返回数量最多的类别
        return max(label_list, key=label_list.count)

    # 选择最优特征
    best_feature_idx = choose_best_feature(data)
    best_feature_label = labels[best_feature_idx]

    # 构建树
    decision_tree = {best_feature_label: {}}
    # 删除已使用的特征
    del (labels[best_feature_idx])

    # 遍历最优特征的所有取值，递归构建子树
    feature_values = [sample[best_feature_idx] for sample in data]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        decision_tree[best_feature_label][value] = create_tree(
            split_data(data, best_feature_idx, value), sub_labels
        )
    return decision_tree


# ===================== 6. 决策树可视化（使用matplotlib） =====================
def plot_tree_matplotlib(tree, ax=None, parent_pos=None, edge_label=None, pos=(0.5, 1.0), level=0, width=0.25):
    """
    使用matplotlib可视化决策树
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    if isinstance(tree, dict):
        root = list(tree.keys())[0]
        node_name = str(root)

        # 绘制当前节点
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor="#4CAF50", edgecolor="black")
        ax.text(pos[0], pos[1], node_name, fontsize=14, ha='center', va='center',
                bbox=bbox_props, fontweight='bold')

        if parent_pos is not None:
            ax.annotate(edge_label, xy=pos, xytext=(parent_pos[0], parent_pos[1]),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                       fontsize=12, ha='center', va='center')

        # 计算子节点位置
        num_children = len(tree[root])
        children = list(tree[root].items())
        y_offset = 0.18

        if num_children == 1:
            child_pos = (pos[0], pos[1] - y_offset)
            plot_tree_matplotlib(children[0][1], ax, pos, str(children[0][0]), child_pos, level + 1, width)
        else:
            start_x = pos[0] - width * (num_children - 1) / 2
            for i, (key, value) in enumerate(children):
                child_x = start_x + i * width
                child_pos = (child_x, pos[1] - y_offset)
                plot_tree_matplotlib(value, ax, pos, str(key), child_pos, level + 1, width * 0.6)
    else:
        # 叶子节点
        node_name = f"Result: {tree}"
        bbox_props = dict(boxstyle="ellipse,pad=0.3", facecolor="#FFC107", edgecolor="black")
        ax.text(pos[0], pos[1], node_name, fontsize=13, ha='center', va='center',
                bbox=bbox_props, fontweight='bold')

        if parent_pos is not None:
            ax.annotate(edge_label, xy=pos, xytext=(parent_pos[0], parent_pos[1]),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                       fontsize=12, ha='center', va='center')

    return ax


# ===================== 7. 预测函数 =====================
def predict(tree, labels, test_sample):
    """
    单样本预测
    :param tree: 决策树
    :param labels: 特征标签
    :param test_sample: 测试样本
    :return: 预测结果
    """
    root = list(tree.keys())[0]
    feature_idx = labels.index(root)
    sub_tree = tree[root][test_sample[feature_idx]]
    if isinstance(sub_tree, dict):
        return predict(sub_tree, labels, test_sample)
    else:
        return sub_tree


# ===================== 主函数 =====================
if __name__ == '__main__':
    # 1. 加载数据
    data, origin_labels = create_data()
    labels = origin_labels.copy()

    # 2. 构建决策树
    decision_tree = create_tree(data, labels)
    print("=== Built ID3 Decision Tree ===")
    print(decision_tree)

    # 3. 可视化决策树
    try:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("ID3 Decision Tree Visualization", fontsize=16, fontweight='bold')
        plot_tree_matplotlib(decision_tree, ax)
        plt.tight_layout()
        plt.savefig("id3_decision_tree.png", dpi=150, bbox_inches='tight', facecolor='white')
        print("\nDecision tree visualization saved: id3_decision_tree.png")
    except Exception as e:
        print(f"\nVisualization failed: {e}")

    # 4. 测试预测
    print("\n=== Test Predictions ===")
    test_cases = [
        ['坏', '否', '否'],  # 预期：低
        ['好', '是', '是'],  # 预期：高
        ['坏', '是', '否']  # 预期：低
    ]
    for case in test_cases:
        res = predict(decision_tree, origin_labels.copy(), case)
        print(f"Weather: {case[0]}, Weekend: {case[1]}, Promotion: {case[2]} -> Sales: {res}")