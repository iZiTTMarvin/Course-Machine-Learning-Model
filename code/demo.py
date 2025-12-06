"""
机器学习综合案例实践演示脚本
展示所有算法的完整流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
import os

# 导入自定义算法
from data_preprocessing import DataPreprocessor
from 聚类.kmeans import MyKMeans
from 聚类.dbscan import MyDBSCAN
from 分类.knn import MyKNN
from 分类.naive_bayes import MyGaussianNB
from 对比.clustering_comparison import ClusteringComparison
from 对比.all_classification_comparison import AllClassificationComparison
from config import OUTPUT_DIR

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主演示函数"""
    print("="*80)
    print("机器学习综合案例实践演示 - 完整版")
    print("="*80)
    
    # 1. 数据预处理
    print("\n【步骤1】数据预处理...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    X, y = data['X_normalized'], data['y_encoded']
    label_mapping = data['label_mapping']
    
    print(f"\n数据预处理完成:")
    print(f"  - 特征矩阵形状: {X.shape}")
    print(f"  - 标签向量形状: {y.shape}")
    print(f"  - 类别数: {len(np.unique(y))}")
    
    # 2. K-Means聚类（带可视化）
    print("\n【步骤2】K-Means聚类（生成可视化图）...")
    kmeans = MyKMeans()
    kmeans_labels = kmeans.fit_predict(X)
    
    kmeans_ari = adjusted_rand_score(y, kmeans_labels)
    kmeans_nmi = normalized_mutual_info_score(y, kmeans_labels)
    
    print(f"K-Means聚类结果:")
    print(f"  - 最终惯性: {kmeans.inertia:.2f}")
    print(f"  - ARI评分: {kmeans_ari:.4f}")
    print(f"  - NMI评分: {kmeans_nmi:.4f}")
    print(f"  - 簇分布: {np.bincount(kmeans_labels)}")
    
    # 生成K-Means可视化
    print("\n  生成K-Means聚类可视化图...")
    kmeans.plot_clusters(X, y, save_path=os.path.join(OUTPUT_DIR, 'K-Means聚类结果.png'))
    
    # 3. DBSCAN聚类（带可视化）
    print("\n【步骤3】DBSCAN聚类（生成可视化图）...")
    dbscan = MyDBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    n_noise = np.sum(dbscan_labels == -1)
    n_clusters = len(np.unique(dbscan_labels)) - (1 if n_noise > 0 else 0)
    
    mask_no_noise = dbscan_labels != -1
    if np.sum(mask_no_noise) > 0:
        dbscan_ari = adjusted_rand_score(y[mask_no_noise], dbscan_labels[mask_no_noise])
        dbscan_nmi = normalized_mutual_info_score(y[mask_no_noise], dbscan_labels[mask_no_noise])
    else:
        dbscan_ari, dbscan_nmi = 0, 0
    
    print(f"DBSCAN聚类结果:")
    print(f"  - 发现簇数: {n_clusters}")
    print(f"  - 噪声点: {n_noise}")
    print(f"  - ARI评分: {dbscan_ari:.4f}")
    print(f"  - NMI评分: {dbscan_nmi:.4f}")
    
    # 生成DBSCAN可视化
    print("\n  生成DBSCAN聚类可视化图...")
    dbscan.plot_clusters(X, y, save_path=os.path.join(OUTPUT_DIR, 'DBSCAN聚类结果.png'))
    
    # 4. 聚类算法对比（生成对比图）
    print("\n【步骤4】聚类算法对比（生成对比图）...")
    clustering_comparison = ClusteringComparison()
    clustering_results = clustering_comparison.compare_algorithms(X, y)
    
    # 5. 数据分割
    print("\n【步骤5】数据分割...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    
    # 6. KNN分类（带可视化）
    print("\n【步骤6】KNN分类（生成可视化图）...")
    knn = MyKNN(k=5)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    
    print(f"KNN分类结果:")
    print(f"  - 准确率: {knn_accuracy:.4f}")
    
    # 生成KNN可视化
    print("\n  生成KNN分类可视化图...")
    knn.plot_results(X_train, y_train, X_test, y_test, knn_predictions, label_mapping,
                    save_path=os.path.join(OUTPUT_DIR, 'KNN分类结果.png'))
    
    # 7. 朴素贝叶斯分类（带可视化）
    print("\n【步骤7】朴素贝叶斯分类（生成可视化图）...")
    gnb = MyGaussianNB()
    gnb.fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)
    
    gnb_accuracy = accuracy_score(y_test, gnb_predictions)
    
    print(f"朴素贝叶斯分类结果:")
    print(f"  - 准确率: {gnb_accuracy:.4f}")
    
    # 生成朴素贝叶斯可视化
    print("\n  生成朴素贝叶斯分类可视化图...")
    gnb.plot_results(X_train, y_train, X_test, y_test, gnb_predictions, label_mapping,
                    save_path=os.path.join(OUTPUT_DIR, '朴素贝叶斯分类结果.png'))
    
    # 8. 完整分类算法对比（生成对比图）
    print("\n【步骤8】完整分类算法对比（生成对比图）...")
    classification_comparison = AllClassificationComparison()
    classification_results = classification_comparison.compare_all_algorithms(X, y, label_mapping)
    
    # 9. 结果总结
    print("\n" + "="*80)
    print("实验结果总结")
    print("="*80)
    
    print(f"\n【聚类算法对比】")
    print(f"  - K-Means ARI: {kmeans_ari:.4f}")
    print(f"  - DBSCAN ARI: {dbscan_ari:.4f}")
    print(f"  - 最佳聚类算法: {'K-Means' if kmeans_ari > dbscan_ari else 'DBSCAN'}")
    
    print(f"\n【分类算法对比】")
    # 从对比结果中获取所有算法的准确率
    for algo_name, result in classification_results.items():
        if 'accuracy' in result:
            print(f"  - {algo_name}准确率: {result['accuracy']:.4f}")
    
    # 找出最佳分类算法
    best_classifier = max(classification_results.items(), key=lambda x: x[1].get('accuracy', 0))
    print(f"  - 最佳分类算法: {best_classifier[0]} ({best_classifier[1]['accuracy']:.4f})")
    
    print(f"\n【整体最佳算法】")
    best_clustering = 'K-Means' if kmeans_ari > dbscan_ari else 'DBSCAN'
    print(f"  - 最佳聚类算法: {best_clustering}")
    print(f"  - 最佳分类算法: {best_classifier[0]}")
    
    print(f"\n【生成的可视化文件】")
    print(f"  所有结果文件已保存到: {OUTPUT_DIR}")
    print(f"  1. K-Means聚类结果.png")
    print(f"  2. DBSCAN聚类结果.png")
    print(f"  3. 聚类算法对比.png")
    print(f"  4. KNN分类结果.png")
    print(f"  5. 朴素贝叶斯分类结果.png")
    print(f"  6. 所有分类算法混淆矩阵对比.png")
    print(f"  7. 所有分类算法准确率对比.png")
    print(f"  8. 分类算法详细性能指标对比.png")
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)

if __name__ == "__main__":
    main()