"""
机器学习综合案例实践演示脚本
展示所有算法的完整流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns

# 导入自定义算法
from data_preprocessing import DataPreprocessor
from kmeans import MyKMeans
from knn import MyKNN
from naive_bayes import MyGaussianNB

def main():
    """主演示函数"""
    print("="*60)
    print("机器学习综合案例实践演示")
    print("="*60)
    
    # 1. 数据预处理
    print("\n1. 数据预处理...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    X, y = data['X_normalized'], data['y_encoded']
    label_mapping = data['label_mapping']
    
    print(f"\n数据预处理完成:")
    print(f"- 特征矩阵形状: {X.shape}")
    print(f"- 标签向量形状: {y.shape}")
    print(f"- 类别数: {len(np.unique(y))}")
    
    # 2. K-Means聚类
    print("\n2. K-Means聚类...")
    kmeans = MyKMeans()
    kmeans_labels = kmeans.fit_predict(X)
    
    kmeans_ari = adjusted_rand_score(y, kmeans_labels)
    kmeans_nmi = normalized_mutual_info_score(y, kmeans_labels)
    
    print(f"K-Means聚类结果:")
    print(f"- 最终惯性: {kmeans.inertia:.2f}")
    print(f"- ARI评分: {kmeans_ari:.4f}")
    print(f"- NMI评分: {kmeans_nmi:.4f}")
    print(f"- 簇分布: {np.bincount(kmeans_labels)}")
    
    # 3. DBSCAN聚类对比
    print("\n3. DBSCAN聚类对比...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
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
    print(f"- 发现簇数: {n_clusters}")
    print(f"- 噪声点: {n_noise}")
    print(f"- ARI评分: {dbscan_ari:.4f}")
    print(f"- NMI评分: {dbscan_nmi:.4f}")
    
    # 4. 数据分割
    print("\n4. 数据分割...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 5. KNN分类
    print("\n5. KNN分类...")
    knn = MyKNN(k=5)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    
    print(f"KNN分类结果:")
    print(f"- 准确率: {knn_accuracy:.4f}")
    
    # 6. 朴素贝叶斯分类
    print("\n6. 朴素贝叶斯分类...")
    gnb = MyGaussianNB()
    gnb.fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)
    
    gnb_accuracy = accuracy_score(y_test, gnb_predictions)
    
    print(f"朴素贝叶斯分类结果:")
    print(f"- 准确率: {gnb_accuracy:.4f}")
    
    # 7. 结果总结
    print("\n" + "="*60)
    print("实验结果总结")
    print("="*60)
    
    print(f"\n聚类算法对比:")
    print(f"- K-Means ARI: {kmeans_ari:.4f}")
    print(f"- DBSCAN ARI: {dbscan_ari:.4f}")
    print(f"- 最佳聚类算法: {'K-Means' if kmeans_ari > dbscan_ari else 'DBSCAN'}")
    
    print(f"\n分类算法对比:")
    print(f"- KNN准确率: {knn_accuracy:.4f}")
    print(f"- 朴素贝叶斯准确率: {gnb_accuracy:.4f}")
    print(f"- 最佳分类算法: {'KNN' if knn_accuracy > gnb_accuracy else '朴素贝叶斯'}")
    
    print(f"\n整体最佳算法:")
    best_clustering = 'K-Means' if kmeans_ari > dbscan_ari else 'DBSCAN'
    best_classification = 'KNN' if knn_accuracy > gnb_accuracy else '朴素贝叶斯'
    print(f"- 最佳聚类算法: {best_clustering}")
    print(f"- 最佳分类算法: {best_classification}")
    
    print(f"\n实验完成！所有结果文件已保存到 ../result/")
    print("="*60)

if __name__ == "__main__":
    main()