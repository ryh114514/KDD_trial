import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
'''
废案，无法按图像识别处理，没有这么多数据以生成多张图像，也没有数据标注
'''
# 加载数据
dataset = np.loadtxt('聚类分析作业\\dataset\\data-8-2-1000.txt').astype(np.float32)
print(dataset.shape)

# 自定义损失函数
def cluster_loss(y_true, y_pred):
    """
    y_true: 输入数据点
    y_pred: 重构的数据点
    """
    # 计算重构误差
    reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return reconstruction_loss

def autoencoder_model(input_dim, encoding_dim=8):
    # 编码器
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(encoding_dim, activation='relu')  # 编码后的特征
    ])
    
    # 解码器
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='linear')  # 重构的输入
    ])
    
    # 完整的自编码器
    autoencoder = tf.keras.Sequential([encoder, decoder])
    
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=cluster_loss
    )
    
    return autoencoder, encoder

# 参数设置
input_dim = 2
encoding_dim = 8
num_clusters = 5

# 创建或加载模型
if not os.path.exists('聚类分析作业\\autoencoder_model.h5'):
    autoencoder, encoder = autoencoder_model(input_dim, encoding_dim)
else:
    autoencoder = tf.keras.models.load_model('聚类分析作业\\autoencoder_model.h5', 
                                           custom_objects={'cluster_loss': cluster_loss})
    encoder = autoencoder.layers[0]
    print("Loaded existing model.")

# 训练自编码器
batch_size = 32
epochs = 1000

history = autoencoder.fit(
    dataset, dataset,  # 输入和目标都是数据本身
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

# 保存模型
autoencoder.save('聚类分析作业\\autoencoder_model.h5')

# 获取编码后的特征
encoded_data = encoder.predict(dataset)

# 使用K-means对编码后的特征进行聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(encoded_data)

# 获取聚类中心（在编码空间中）
encoded_centers = kmeans.cluster_centers_

# 将聚类中心解码回原始空间
cluster_centers = autoencoder.layers[1].predict(encoded_centers)

print("聚类中心坐标:")
for i, center in enumerate(cluster_centers):
    print(f"簇 {i}: {center}")

# 可视化聚类结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster_assignments, cmap='viridis', alpha=0.6)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='centers')
plt.title('Autoencoder + K-means results')
plt.colorbar(scatter)
plt.legend()

# 绘制决策边界
plt.subplot(1, 2, 2)
x = np.linspace(dataset[:, 0].min()-1, dataset[:, 0].max()+1, 200)
y = np.linspace(dataset[:, 1].min()-1, dataset[:, 1].max()+1, 200)
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 对网格点进行编码和聚类
grid_encoded = encoder.predict(grid_points)
grid_assignments = kmeans.predict(grid_encoded)
grid_assignments = grid_assignments.reshape(xx.shape)

plt.contourf(xx, yy, grid_assignments, alpha=0.3, cmap='viridis')
plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster_assignments, cmap='viridis', alpha=0.6)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='centers')
plt.title('Decision boundary')
plt.colorbar(scatter)
plt.legend()

plt.tight_layout()
plt.show()
