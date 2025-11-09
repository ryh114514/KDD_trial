import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn import metrics
dataset = np.loadtxt('聚类分析作业\\mid_result\\customers.txt').astype(np.float32)

input_dim = 2  
output_dim = 5

'''
def my_loss(y_true, y_pred):
    dataset_tensor = tf.constant(dataset, dtype=tf.float32)
    
    # 添加softmax确保每行的权重和为1
    weights = tf.nn.softmax(y_pred, axis=1)
    
    epsilon = 1e-8
    cluster_counts = tf.reduce_sum(weights, axis=0)
    
    # 添加空簇惩罚
    empty_cluster_penalty = tf.reduce_sum(tf.maximum(0.0, 50 - cluster_counts)) * 0.1
    
    cluster_counts = tf.maximum(cluster_counts, epsilon)
    cluster_centers = tf.matmul(weights, dataset_tensor, transpose_a=True)  
    cluster_centers = cluster_centers / tf.expand_dims(cluster_counts, axis=1)  
    
    distances = tf.reduce_sum(
        tf.square(tf.expand_dims(dataset_tensor, 1) - tf.expand_dims(cluster_centers, 0)),
        axis=2
    )
    
    center_distances = tf.reduce_sum(
        tf.square(tf.expand_dims(cluster_centers, 1) - tf.expand_dims(cluster_centers, 0)),
        axis=2
    )
    
    # 重新添加簇间距离惩罚
    mask = 1 - tf.eye(tf.shape(center_distances)[0])  # 排除对角线
    inter_cluster_loss = -tf.reduce_mean(center_distances * mask) * 0.5
    
    intra_cluster_loss = tf.reduce_mean(weights * distances)
    
    # 组合所有损失
    total_loss = intra_cluster_loss + inter_cluster_loss + empty_cluster_penalty
    total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.constant(1e6))
    return total_loss
'''
def my_loss(y_true, y_pred):
    dataset_tensor = tf.constant(dataset, dtype=tf.float32)
    
    # 使用softmax确保每行的权重和为1
    weights = tf.nn.softmax(y_pred, axis=1)
    
    epsilon = 1e-8
    cluster_counts = tf.reduce_sum(weights, axis=0)
    
    # 改进的空簇惩罚 - 更温和
    empty_cluster_penalty = tf.reduce_sum(tf.maximum(0.0, 50 - cluster_counts)) * 0.1
    
    cluster_counts = tf.maximum(cluster_counts, epsilon)
    cluster_centers = tf.matmul(weights, dataset_tensor, transpose_a=True)  
    cluster_centers = cluster_centers / tf.expand_dims(cluster_counts, axis=1)  
    
    # 簇内距离
    distances = tf.reduce_sum(
        tf.square(tf.expand_dims(dataset_tensor, 1) - tf.expand_dims(cluster_centers, 0)),
        axis=2
    )
    intra_cluster_loss = tf.reduce_mean(weights * distances)
    
    # 簇间距离 ：鼓励簇间距离大
    center_distances = tf.reduce_sum(
        tf.square(tf.expand_dims(cluster_centers, 1) - tf.expand_dims(cluster_centers, 0)),
        axis=2
    )
    
    mask = 1 - tf.eye(tf.shape(center_distances)[0])  # 排除对角线
    # 修正：使用正号鼓励簇间距离大
    inter_cluster_reward = tf.reduce_mean(center_distances * mask) * 0.01
    
    # 多样性惩罚 - 鼓励权重分布均匀
    #weight_entropy = -tf.reduce_mean(tf.reduce_sum(weights * tf.math.log(weights + epsilon), axis=1))
    #diversity_penalty = weight_entropy * 0.1
    
    # 组合损失：簇内距离小 + 簇间距离大 + 分布均匀 - 空簇惩罚
    total_loss = intra_cluster_loss - inter_cluster_reward*20  + empty_cluster_penalty
    
    return total_loss
def dnn_model(input_dim, output_dim, loss_function='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='sigmoid', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        
        
        tf.keras.layers.Dense(output_dim, activation='linear')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss_function,
        metrics=metrics,
        
    )
    
    return model




# 数据预处理 - 标准化
def preprocess_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

# 主程序
dataset_normalized = preprocess_data(dataset)

if os.path.exists(f'聚类分析作业\\result\\customers\\dnnv1_model_{output_dim}.h5'):  
    model = tf.keras.models.load_model(f'聚类分析作业\\result\\customers\\dnnv1_model_{output_dim}.h5', 
                                      custom_objects={'cluster_loss': my_loss}, 
                                      compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                 loss=my_loss)
else:
    model = dnn_model(input_dim, output_dim, loss_function=my_loss, optimizer='adam', metrics=['accuracy'])
# 重新训练模型

model.summary()

# 训练模型
print("开始训练...")
history = model.fit(
    dataset_normalized,  
    np.zeros((dataset_normalized.shape[0], output_dim)),  
    epochs=1000,
    batch_size=1000,
    verbose=1,
    shuffle=True
)

model.save(f'聚类分析作业\\result\\customers\\dnnv1_model_{output_dim}.h5')
# 预测
predictions = model.predict(dataset_normalized)
# 使用argmax得到硬分配
cluster_assignments = np.argmax(predictions, axis=1)

# 计算每个簇的点数
unique, counts = np.unique(cluster_assignments, return_counts=True)
print(f"簇分布: {dict(zip(unique, counts))}")

# 可视化结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster_assignments, cmap='tab10')
plt.title('clustering results')
plt.colorbar()

plt.subplot(1, 3, 2)
# 显示权重分布
weights = tf.nn.softmax(predictions, axis=1).numpy()

for i in range(output_dim):
    plt.hist(weights[:, i], alpha=0.5, label=f'cluster {i}')
plt.title('distribution of cluster weights')
plt.legend()

plt.subplot(1, 3, 3)
# 训练损失
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# 生成决策边界
x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_normalized = (grid_points - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

grid_predictions = model.predict(grid_points_normalized)

grid_assignments = np.argmax(grid_predictions, axis=1)


weights = tf.nn.softmax(predictions, axis=1).numpy()
cluster_centers = []
for i in range(output_dim):
    if np.sum(weights[:, i]) > 0:
        center = np.average(dataset, axis=0, weights=weights[:, i])
        cluster_centers.append(center)
    else:
        # 如果某个簇没有点，使用随机点作为中心
        center = np.mean(dataset, axis=0)
        cluster_centers.append(center)

cluster_centers = np.array(cluster_centers)

# 重新绘制带聚类中心的图
plt.figure(figsize=(12, 10))

# 绘制决策边界
plt.contourf(xx, yy, grid_assignments.reshape(xx.shape), alpha=0.3, cmap='tab10')

# 绘制数据点
scatter = plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster_assignments, cmap='tab10', 
                     edgecolors='black', alpha=0.7, s=30)

# 绘制聚类中心
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=range(output_dim), 
           cmap='tab10', marker='*', s=300, edgecolors='red', linewidth=2, 
           label='Cluster Centers')

# 添加聚类中心标签
for i, center in enumerate(cluster_centers):
    plt.annotate(f'Center {i}', xy=center, xytext=(center[0]+0.5, center[1]+0.5),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=12, color='darkred')

plt.title('Clustering Results with Centers')
plt.colorbar(scatter)
plt.legend()
plt.show()
silhouette_score = metrics.silhouette_score(dataset, cluster_assignments)
calinski_harabasz_score = metrics.calinski_harabasz_score(dataset, cluster_assignments)
davies_bouldin_score = metrics.davies_bouldin_score(dataset, cluster_assignments)

print("\n聚类评估指标:")
print(f"轮廓系数: {silhouette_score:.4f}")
print(f"Calinski-Harabasz指数: {calinski_harabasz_score:.4f}")
print(f"Davies-Bouldin指数: {davies_bouldin_score:.4f}")