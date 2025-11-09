import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_kernels
import seaborn as sns
from tool import visualize_clusters
# 1.设置中文显示
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 2.读取数据
#SP500 = np.genfromtxt('chapter8\\data\\SP500array.csv', delimiter=',').T
#nStock = len(SP500[:, 0])
# 3.标准化数据
#X = (SP500 - np.mean(SP500, axis=1).reshape(-1, 1)) / np.std(SP500, axis=1).reshape(-1, 1)
X=np.loadtxt('聚类分析作业\\mid_result\\SP500_1.txt')
# 4.使用t-SNE降维
Y = X#manifold.TSNE(n_components=2, random_state=0).fit_transform(X)
# 5.计算谱聚类和K-means的三个评估指标
best_score = float('-inf')
best_n_clusters = 0
best_labels_spectral = None
best_labels_kmeans = None
X = np.loadtxt('聚类分析作业\\mid_result\\SP500_1.txt')
Y=np.loadtxt('聚类分析作业\\mid_result\\SP500_2.txt')
dist_matrix = np.loadtxt('聚类分析作业\\mid_result\\SP500_2_matrix.txt')
for n_clusters in range(2, 11):
    # 谱聚类
    affinity = dist_matrix
    labels_spectral = spectral_clustering(affinity=affinity,
                                          n_clusters=n_clusters)
    
    silhouette = silhouette_score(dist_matrix, labels_spectral, metric='precomputed')
   
    # K-means聚类
    '''
    labels_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(Y)
    db_score_kmeans = davies_bouldin_score(Y, labels_kmeans)
    '''
    if silhouette > best_score:
        best_score = silhouette
        best_n_clusters = n_clusters
        best_labels_spectral = labels_spectral
    
# 6.使用最佳聚类数进行聚类
print(f"最佳聚类数: {best_n_clusters}")
silhouette_best = silhouette_score(dist_matrix, best_labels_spectral, metric='precomputed')
'''
calinski_harabasz_best = calinski_harabasz_score(Y, best_labels_spectral)
davies_bouldin_best = davies_bouldin_score(Y, best_labels_spectral)
'''
print("\n谱聚类评估指标:")
print(f"轮廓系数: {silhouette_best:.3f}")
#print(f"Calinski-Harabasz指数: {calinski_harabasz_best:.3f}")
#print(f"Davies-Bouldin指数: {davies_bouldin_best:.3f}")
# 7.绘制谱聚类结果
visualize_clusters(X,best_labels_spectral)
visualize_clusters(Y,best_labels_spectral)
'''
plt.figure(figsize=(14, 9))
sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=best_labels_spectral,
                alpha=1, palette="Set2",s=100)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("x", fontsize=40)
plt.ylabel("y", fontsize=40)
plt.legend(fontsize=20,bbox_to_anchor=(1, 1))
plt.title("谱聚类结果", fontsize=40)
plt.show()
'''
# 8.使用最佳聚类数的K-means聚类
'''
best_labels_kmeans = KMeans(n_clusters=best_n_clusters,
                            random_state=0).fit_predict(Y)


silhouette_kmeans = silhouette_score(Y, best_labels_kmeans)
calinski_harabasz_kmeans = calinski_harabasz_score(Y, best_labels_kmeans)
davies_bouldin_kmeans = davies_bouldin_score(Y, best_labels_kmeans)

print("\nK-means评估指标:")
print(f"轮廓系数: {silhouette_kmeans:.3f}")
print(f"Calinski-Harabasz指数: {calinski_harabasz_kmeans:.3f}")
print(f"Davies-Bouldin指数: {davies_bouldin_kmeans:.3f}")
# 9.绘制K-means聚类结果
plt.figure(figsize=(14, 9))
sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=best_labels_kmeans,
                alpha=1, palette="Set2",s=100)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("x", fontsize=40)
plt.ylabel("y", fontsize=40)
plt.legend(fontsize=20,bbox_to_anchor=(1, 1))
plt.title("K-means聚类结果", fontsize=40)
plt.show()
'''