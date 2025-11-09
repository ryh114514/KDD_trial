import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# 1.设置绘图字体
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 2.导入数据集
dataset = np.loadtxt('聚类分析作业\\mid_result\\customers.txt').astype(np.float64)
X = dataset[:, :2]
# 3.使用肘部法则确定聚类数量
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
# 4.绘制肘部法则图
plt.figure(figsize=(10, 8))
plt.rc('font', size=30)
plt.plot(range(1, 11), wcss, marker="o",linewidth=2,c="dodgerblue")
plt.plot(5,44448.45544793371,c="r",marker="X",markersize=20)
plt.xlabel('簇数量',fontsize=30)
plt.ylabel('距离平方和',fontsize=30)
plt.xticks(fontsize = 30)
plt.ticklabel_format(style="sci",scilimits=(-1,2),axis="y")
plt.yticks(fontsize = 30)
plt.show()
# 5.K-Means聚类
kmeans = KMeans(n_clusters=3, init='k-means++',
                max_iter=300, random_state=0)
Y_Kmeans = kmeans.fit_predict(X)
# 6.创建用于绘制边界的网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                     np.linspace(y_min, y_max, 1000))
# 7.预测聚类标签
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# 8.绘制聚类边界和结果
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.1,cmap='Set2')
colors = ['grey', 'salmon', 'peru', 'palegreen', 'steelblue']
sns.scatterplot(x=X[:, 0], y = X[:, 1],
                hue=Y_Kmeans,palette="Set2")
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], s=100,
            c='tab:red', label='质心')
plt.xlabel('',fontsize=30)
plt.ylabel('',fontsize=30)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.legend(fontsize=20)
plt.show()
# 计算聚类评估指标
from sklearn import metrics
silhouette_score = metrics.silhouette_score(X, Y_Kmeans)
calinski_harabasz_score = metrics.calinski_harabasz_score(X, Y_Kmeans)
davies_bouldin_score = metrics.davies_bouldin_score(X, Y_Kmeans)

print("\nkmeans聚类评估指标:")
print(f"轮廓系数: {silhouette_score:.4f}")
print(f"Calinski-Harabasz指数: {calinski_harabasz_score:.4f}")
print(f"Davies-Bouldin指数: {davies_bouldin_score:.4f}")