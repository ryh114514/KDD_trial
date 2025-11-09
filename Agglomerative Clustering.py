import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from tool import visualize_clusters

# 1.设置绘图字体
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 2.绘制树状层次图的函数
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, **kwargs)
# 3.导入数据并分离特征和标签
#data = np.loadtxt('chapter8\\data\\seeds_dataset.txt')
#X = data[:, :-1]
X=np.loadtxt('聚类分析作业\\mid_result\\SP500_2.txt')
Y=np.loadtxt('聚类分析作业\\mid_result\\SP500_1.txt')
dist_matrix = np.loadtxt('聚类分析作业\\mid_result\\SP500_2_matrix.txt')
# 4.从数据中拟合层次聚类并绘制树状图
model = AgglomerativeClustering(linkage='average',metric='precomputed',distance_threshold=0,
                                n_clusters=None)
model = model.fit(dist_matrix)#用距离矩阵进行聚类
plt.figure(figsize=(10, 10))
plot_dendrogram(model, truncate_mode='level', p=2)
plt.xlabel("簇中的样本数量",fontsize=30)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 30)
plt.show()
# 5.用PCA进行降维以便于可视化
'''
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
'''
X_pca = X
# 6.层次聚类
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

'''
print("调整兰德系数值为{:.3f}".
      format( adjusted_rand_score(data[:, -1], labels)))
'''
'''
plt.figure(figsize=(10, 10))
sns.scatterplot(x = X_pca[:, 0], y = X_pca[:, 1],
                hue=labels, palette="Set2")
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.legend(fontsize=20)
plt.show()
'''
#折线图
visualize_clusters(Y,labels)
visualize_clusters(X,labels)
# 计算并输出三个评估指标
silhouette = silhouette_score(dist_matrix, labels, metric='precomputed')
#不支持自定义距离函数
#calinski_harabasz = calinski_harabasz_score(X, labels)
#davies_bouldin = davies_bouldin_score(X, labels)

print("\n层次聚类评估指标:")
print(f"轮廓系数: {silhouette:.3f}")
#print(f"Calinski-Harabasz指数: {calinski_harabasz:.3f}")
#print(f"Davies-Bouldin指数: {davies_bouldin:.3f}")