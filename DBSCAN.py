import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from tool import visualize_clusters
# 1.设置绘图字体
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
'''
# 2.导入数据集、预处理
df = pd.read_csv('chapter8\\data\\CC GENERAL.csv')
df = df.drop(['CUST_ID'], axis=1)
df = df.dropna()
X = pd.DataFrame(StandardScaler().fit_transform(df))
# 3.PCA降维
X = np.asarray(X)
pca = PCA(n_components=2, random_state=24)
X_pca = pca.fit_transform(X)
'''
X_pca=np.loadtxt('聚类分析作业\mid_result\GENERAL.txt').astype(np.float32)
X = np.loadtxt('聚类分析作业\\mid_result\\SP500_1.txt')
Y=np.loadtxt('聚类分析作业\\mid_result\\SP500_2.txt')
dist_matrix = np.loadtxt('聚类分析作业\\mid_result\\SP500_2_matrix.txt')
# 4.实施DBSCAN聚类
dbscan = DBSCAN(eps=0.35,          
                min_samples=10,  
                metric='precomputed')
y_dbscan = dbscan.fit_predict(dist_matrix)
'''
dbscan = DBSCAN(eps=0.7, min_samples=8)
y_dbscan = dbscan.fit_predict(X_pca)
'''
# 5.实施K-means聚类
kmeans = KMeans(n_clusters=3, random_state=24)
y_kmeans = kmeans.fit_predict(X_pca)
# 6.DBSCAN算法结果可视化
'''
plt.figure(figsize=(14, 9))
sns.scatterplot(x=X_pca[:, 0],y=X_pca[:, 1],hue=y_dbscan,palette="Set2",s=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("x", fontsize=40)
plt.ylabel("y", fontsize=40)
plt.legend(fontsize = 20)
plt.title("DBSCAN算法聚类结果", fontsize=40)
plt.show()
'''
# 7.K-means算法可视化
'''
plt.figure(figsize=(14, 9))
sns.scatterplot(x=X_pca[:, 0],y=X_pca[:, 1],hue=y_kmeans,palette="Set2",s=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("x", fontsize=40)
plt.ylabel("y", fontsize=40)
plt.legend(fontsize = 20)
plt.title("K-means算法聚类结果", fontsize=40)
plt.show()
'''
# 计算评估指标
silhouette_dbscan = metrics.silhouette_score(dist_matrix, y_dbscan, metric='precomputed')
'''
silhouette_kmeans = metrics.silhouette_score(X_pca, y_kmeans)
calinski_harabasz_dbscan = metrics.calinski_harabasz_score(X_pca, y_dbscan)
calinski_harabasz_kmeans = metrics.calinski_harabasz_score(X_pca, y_kmeans)
davies_bouldin_dbscan = metrics.davies_bouldin_score(X_pca, y_dbscan)
davies_bouldin_kmeans = metrics.davies_bouldin_score(X_pca, y_kmeans)
'''

print("\nDBSCAN算法评估指标:")
print(f"轮廓系数: {silhouette_dbscan:.3f}")
visualize_clusters(X,y_dbscan)
visualize_clusters(Y,y_dbscan)
'''
print(f"Calinski-Harabasz指数: {calinski_harabasz_dbscan:.3f}")
print(f"Davies-Bouldin指数: {davies_bouldin_dbscan:.3f}")

print("\nK-means算法评估指标:")
print(f"轮廓系数: {silhouette_kmeans:.3f}")
print(f"Calinski-Harabasz指数: {calinski_harabasz_kmeans:.3f}")
print(f"Davies-Bouldin指数: {davies_bouldin_kmeans:.3f}")
'''
