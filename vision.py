import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import warnings

warnings.filterwarnings('ignore')

def vision(predict):
    data = pd.read_csv('/home/sylai/code/paper17/code1/data/csv/test.csv')
    data['Pre'] = 0
    print(data.shape[0], len(predict))
    for i in range(data.shape[0]):
        data["Pre"][i] = predict[i]

    data.to_csv('/home/sylai/code/paper17/code1/data/csv/test_v1.csv', index=None)



def visualize_3d_vectors(vectors, labels, title="test"):
    """
    可视化三维向量

    参数:
    vectors: numpy数组，形状为 (n_samples, 3)
    labels: 列表或数组，形状为 (n_samples,)，包含每个样本的类别标签
    title: 图表标题
    """

    if len(vectors) != len(labels):
        raise ValueError("向量数量和标签数量必须相同")

    if vectors.shape[1] != 3:
        raise ValueError("向量必须是3维的")

    # 获取唯一的类别
    unique_labels = np.unique(labels)
    if len(unique_labels) != 3:
        print(f"警告: 期望3个类别，但找到了{len(unique_labels)}个类别")

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 为每个类别设置不同的颜色和标记
    colors = ['r', 'g', 'b']
    markers = ['o', 's', '^']

    # 为每个类别绘制散点
    for i, label in enumerate(unique_labels):
        # 获取当前类别的向量
        class_vectors = vectors[labels == label]

        # 绘制散点
        ax.scatter(class_vectors[:, 0],
                   class_vectors[:, 1],
                   class_vectors[:, 2],
                   c=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   label=f'Class {label}',
                   s=50, alpha=0.7)

    # 设置坐标轴标签
    ax.set_xlabel('X - Dimension 1', fontsize=12)
    ax.set_ylabel('Y - Dimension 2', fontsize=12)
    ax.set_zlabel('Z - Dimension 3', fontsize=12)

    # 设置标题
    ax.set_title(title, fontsize=14, pad=20)

    # 添加图例
    ax.legend(loc='best')

    # 设置视角（可选，可以调整以获得更好的视图）
    ax.view_init(elev=20, azim=45)

    # 添加网格
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print("可视化统计信息:")
    print(f"总样本数: {len(vectors)}")
    for i, label in enumerate(unique_labels):
        class_count = np.sum(labels == label)
        print(f"类别 {label}: {class_count} 个样本")



def visualize_vectors_2d(vectors, labels, method='pca', title="2D Model Output Visualization"):
    """
    降维可视化2D向量

    参数:
    vectors: numpy数组，形状为 (n_samples, 3)
    labels: 列表或数组，形状为 (n_samples,)，包含每个样本的类别标签
    method: 降维方法，可选 'pca', 'tsne', 'isomap'
    title: 图表标题
    """
    if len(vectors) != len(labels):
        raise ValueError("向量数量和标签数量必须相同")

    if vectors.shape[1] != 3:
        raise ValueError("向量必须是3维的")

    # 获取唯一的类别
    unique_labels = np.unique(labels)

    # 选择降维方法
    if method.lower() == 'pca':
        # PCA降维
        reducer = PCA(n_components=2)
        vectors_2d = reducer.fit_transform(vectors)
        method_name = "PCA"

        # 解释方差比
        explained_var = reducer.explained_variance_ratio_
        print(f"PCA解释方差比: PC1: {explained_var[0]:.2%}, PC2: {explained_var[1]:.2%}")
        print(f"累计解释方差: {sum(explained_var):.2%}")

    elif method.lower() == 'tsne':
        # t-SNE降维（适合可视化）
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors) - 1))
        vectors_2d = reducer.fit_transform(vectors)
        method_name = "t-SNE"

    elif method.lower() == 'isomap':
        # Isomap降维（保持流形结构）
        n_neighbors = min(10, len(vectors) - 1)
        reducer = Isomap(n_components=2, n_neighbors=n_neighbors)
        vectors_2d = reducer.fit_transform(vectors)
        method_name = "Isomap"

    else:
        raise ValueError(f"不支持的降维方法: {method}")

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 为每个类别设置不同的颜色和标记
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']

    # 为每个类别绘制散点
    for i, label in enumerate(unique_labels):
        # 获取当前类别的向量
        class_mask = labels == label
        class_vectors_2d = vectors_2d[class_mask]

        # 绘制散点
        ax.scatter(class_vectors_2d[:, 0],
                   class_vectors_2d[:, 1],
                   c=[colors[i]],
                   marker=markers[i % len(markers)],
                   label=f'Class {label}',
                   s=80, alpha=0.7, edgecolors='w', linewidth=0.5)

    # 设置坐标轴标签
    ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
    ax.set_ylabel(f'{method_name} Component 2', fontsize=12)

    # 设置标题
    ax.set_title(f'{title} ({method_name})', fontsize=14, pad=20)

    # 添加图例
    ax.legend(loc='best')

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加中心点
    mean_point = vectors_2d.mean(axis=0)
    ax.scatter(mean_point[0], mean_point[1], c='red', marker='x', s=200,
               label='Mean Point', zorder=5)

    plt.tight_layout()
    plt.show()

    return vectors_2d


def visualize_all_methods(vectors, labels, title="Model Output Visualization"):
    """
    同时显示多种降维方法的结果
    """
    methods = ['pca', 'tsne', 'isomap']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for idx, method in enumerate(methods):
        ax = axes[idx]

        # 降维
        if method == 'pca':
            reducer = PCA(n_components=2)
            vectors_2d = reducer.fit_transform(vectors)
            method_name = "PCA"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors) - 1))
            vectors_2d = reducer.fit_transform(vectors)
            method_name = "t-SNE"
        else:  # isomap
            n_neighbors = min(10, len(vectors) - 1)
            reducer = Isomap(n_components=2, n_neighbors=n_neighbors)
            vectors_2d = reducer.fit_transform(vectors)
            method_name = "Isomap"

        # 绘制每个类别
        for i, label in enumerate(unique_labels):
            class_mask = labels == label
            class_vectors_2d = vectors_2d[class_mask]

            ax.scatter(class_vectors_2d[:, 0], class_vectors_2d[:, 1],
                       c=[colors[i]], s=60, alpha=0.7, label=f'Class {label}')

        ax.set_title(f'{method_name}', fontsize=12)
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='best')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()