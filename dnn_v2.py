import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn import metrics
dataset = np.loadtxt('聚类分析作业\\mid_result\\customers.txt').astype(np.float32)
print(dataset.shape)
# 参数设置
input_dim = 2  
output_dim = 3
# 自定义损失函数
def cluster_loss(y_true, y_pred):
    """
    y_true: 真实标签（这里用不到，设为与dataset相同形状）
    y_pred: 模型输出的聚类中心，形状为 (batch_size, num_clusters * 2)
    """
    
    cluster_centers = tf.reshape(y_pred, [-1, output_dim, 2])  # (batch_size, num_clusters, 2)
    
    # 计算每个点到所有聚类中心的距离
    dataset_tensor = tf.constant(dataset, dtype=tf.float32)  # (1000, 2)
    
    # 扩展维度以便广播计算
    dataset_expanded = tf.expand_dims(dataset_tensor, 1)  # (1000, 1, 2)
    centers_expanded = tf.expand_dims(cluster_centers, 0)  # (1, batch_size, num_clusters, 2)
    
    # 计算距离矩阵
    distances = tf.reduce_sum(
        tf.square(dataset_expanded - centers_expanded),
        axis=3
    )  # 形状: (1000, batch_size, num_clusters)
    
    # 找到每个点的最近聚类中心
    min_distances = tf.reduce_min(distances, axis=2)  # (1000, batch_size)
    
    # 计算内部损失（所有点到最近聚类中心的距离和）
    inter_loss = tf.reduce_mean(min_distances)
    '''
    # 计算聚类中心之间的距离
    center_distances = tf.reduce_sum(
        tf.square(tf.expand_dims(cluster_centers, 2) - tf.expand_dims(cluster_centers, 1)),
        axis=3
    )  # 形状: (batch_size, num_clusters, num_clusters)
    
    # 创建掩码，排除对角线元素（自身与自身的距离）
    mask = 1 - tf.eye(num_clusters)
    
    # 使用常数除以距离的形式
    # 添加一个小的常数防止除以零
    epsilon = 1e-6
    constant = 1.0  # 常数
    inv_distances = constant / (center_distances + epsilon)
    
    # 计算逆距离的平均值作为损失
    out_ut_loss = tf.reduce_mean(inv_distances * mask) * 0.1  # 可以调整权重
    
    # 组合损失
    '''
    total_loss = inter_loss #+ out_ut_loss十分抽象，一旦一个中心没有点，则该中心倾向于被扔的远远的
    return total_loss

def dnn_model(input_dim, num_clusters, loss_function=cluster_loss, optimizer='adam'):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='sigmoid', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation='relu'),
        # 输出层：num_clusters * 2，表示所有聚类中心的坐标
        tf.keras.layers.Dense(num_clusters * 2, activation='linear')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_function
    )
    
    return model



# 创建或加载模型
if not os.path.exists(f'聚类分析作业\\result\\customers\\dnn_model_{output_dim}.h5'):  
    model = dnn_model(input_dim, output_dim)

else:
    model = tf.keras.models.load_model(f'聚类分析作业\\result\\customers\\dnn_model_{output_dim}.h5', 
                                      custom_objects={'cluster_loss': cluster_loss}, 
                                      compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                 loss=cluster_loss)
    print("Loaded existing model.")

model.summary()

# 训练模型
# 由于直接输出聚类中心，输入可以是任意值，这里用全0
dummy_input = np.zeros((1, input_dim), dtype=np.float32)

# 训练循环
for epoch in range(30000):
    loss = model.train_on_batch(dummy_input, np.zeros((1, dataset.shape[0], 2), dtype=np.float32))
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

model.save(f'聚类分析作业\\result\\customers\\dnn_model_{output_dim}.h5')

# 获取聚类中心
centers_output = model.predict(dummy_input)
cluster_centers = centers_output.reshape(output_dim, 2)

print("聚类中心坐标:")
for i, center in enumerate(cluster_centers):
    print(f"簇 {i}: {center}")

# 分配每个点到最近的聚类中心
def assign_clusters(data, centers):
    distances = np.sqrt(((data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)

cluster_assignments = assign_clusters(dataset, cluster_centers)

# 可视化聚类结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster_assignments, cmap='viridis', alpha=0.6)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='centers')
plt.title('DNN results')
plt.colorbar(scatter)
plt.legend()

# 绘制决策边界
plt.subplot(1, 2, 2)
x = np.linspace(dataset[:, 0].min()-1, dataset[:, 0].max()+1, 200)
y = np.linspace(dataset[:, 1].min()-1, dataset[:, 1].max()+1, 200)
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]

grid_assignments = assign_clusters(grid_points, cluster_centers)
grid_assignments = grid_assignments.reshape(xx.shape)

plt.contourf(xx, yy, grid_assignments, alpha=0.3, cmap='viridis')
plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster_assignments, cmap='viridis', alpha=0.6)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='centers')
plt.title('boundary')
plt.colorbar(scatter)
plt.legend()

plt.tight_layout()
plt.show()
silhouette_score = metrics.silhouette_score(dataset, cluster_assignments)
calinski_harabasz_score = metrics.calinski_harabasz_score(dataset, cluster_assignments)
davies_bouldin_score = metrics.davies_bouldin_score(dataset, cluster_assignments)

print("\n聚类评估指标:")
print(f"轮廓系数: {silhouette_score:.4f}")
print(f"Calinski-Harabasz指数: {calinski_harabasz_score:.4f}")
print(f"Davies-Bouldin指数: {davies_bouldin_score:.4f}")