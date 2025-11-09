import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.gridspec as gridspec
from tool import visualize_clusters
# 1. 设置中文显示
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 2. 读取数据
#X = np.loadtxt("聚类分析作业\mid_result\GENERAL.txt")
X = np.loadtxt('聚类分析作业\\mid_result\\SP500_1.txt')
Y=np.loadtxt('聚类分析作业\\mid_result\\SP500_2.txt')
dist_matrix = np.loadtxt('聚类分析作业\\mid_result\\SP500_2_matrix.txt')
# 3. 应用OPTICS聚类
clust = OPTICS(min_samples=5, xi=0.0015, min_cluster_size=0.05,metric='precomputed')
clust.fit(dist_matrix)
# 4. 应用DBSCAN聚类
'''
labels_070 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_, eps=0.7)
labels_030 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_, eps=0.3)
'''
# 5. 绘图

plt.figure(figsize=(16, 10))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
'''
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])
'''
# 6. 可达性图
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
space = np.arange(len(X))
colors = ['palegreen', "tab:blue", "olive", "salmon",
          'peru', 'steelblue', 'magenta', 'cyan']
for klass, color in zip(range(0, 8), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color)
ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
ax1.plot(space, np.full_like(space, 0.7, dtype=float), "k-", alpha=0.5)
ax1.plot(space, np.full_like(space, 0.3, dtype=float), "k-.", alpha=0.5)
ax1.set_ylabel("可达距离", fontsize=20)
ax1.set_title("OPTICS可达性图", fontsize=30)
ax1.tick_params(labelsize=20)
plt.show()
visualize_clusters(X,labels)
visualize_clusters(Y,labels)
# 7. OPTICS聚类结果可视化
'''
for klass, color in zip(range(0, 8), colors):
    Xk = X[clust.labels_ == klass]
    ax2.scatter(Xk[:, 0], Xk[:, 1], c=color, alpha=0.5, marker='o')
ax2.scatter(x=X[clust.labels_ == -1, 0], y=X[clust.labels_ == -1, 1],
            c="gray", alpha=0.3, marker="^", s=10)
ax2.set_title("OPTICS聚类结果", fontsize=20)
ax2.tick_params(labelsize=20)
'''
# 8. DBSCAN聚类结果可视化（eps=0.7）
'''
for klass, color in zip(range(0, 8), colors):
    Xk = X[labels_070 == klass]
    ax3.scatter(Xk[:, 0], Xk[:, 1], c=color, alpha=0.5, marker='o')
ax3.scatter(x=X[labels_070 == -1, 0], y=X[labels_070 == -1, 1],
            c="gray", alpha=0.3, marker="^", s=10)
ax3.set_title("DBSCAN聚类结果 (eps=0.7)", fontsize=20)
ax3.tick_params(labelsize=20)
'''
# 9. DBSCAN聚类结果可视化（eps=0.3）
'''
for klass, color in zip(range(0, 8), colors):
    Xk = X[labels_030 == klass]
    ax4.scatter(Xk[:, 0], Xk[:, 1], c=color, alpha=0.5, marker='o')
ax4.scatter(x=X[labels_030 == -1, 0], y=X[labels_030 == -1, 1],
            c="gray", alpha=0.3, marker="^", s=10)
ax4.set_title("DBSCAN聚类结果 (eps=0.3)", fontsize=20)
ax4.tick_params(labelsize=20)
plt.show()
'''
'''
def print_metrics(X, labels, method_name):
    if len(np.unique(labels)) > 1:  # 确保有多个簇
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        print(f"\n{method_name}聚类评估指标:")
        print(f"轮廓系数: {silhouette:.3f}")
        print(f"Calinski-Harabasz指数: {calinski_harabasz:.3f}")
        print(f"Davies-Bouldin指数: {davies_bouldin:.3f}")
    else:
        print(f"\n{method_name}只识别出一个簇或噪声点，无法计算评估指标")

# 计算三种聚类方法的评估指标
print_metrics(X, clust.labels_, "OPTICS")
print_metrics(X, labels_070, "DBSCAN (eps=0.7)")
print_metrics(X, labels_030, "DBSCAN (eps=0.3)")
'''
print("\noptics聚类评估指标:")
print(f"轮廓系数: {silhouette_score(dist_matrix, clust.labels_, metric='precomputed'):.3f}")