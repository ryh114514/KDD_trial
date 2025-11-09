import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#dataset = np.loadtxt('聚类分析作业\\dataset\\data-8-2-1000.txt').astype(np.float32)

#数据离散化
def Discretization(points, spacing):
    # 确保spacing是元组形式
    if isinstance(spacing, (int, float)):
        spacing = (spacing, spacing)
    
    # 获取点集的边界
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # 计算网格的大小
    grid_size_x = int(np.ceil((max_coords[0] - min_coords[0]) / spacing[0]))
    grid_size_y = int(np.ceil((max_coords[1] - min_coords[1]) / spacing[1]))
    
    # 创建二维网格
    grid = np.zeros((grid_size_y, grid_size_x))
    
    # 计算每个点所在的网格位置
    grid_indices = np.floor((points - min_coords) / spacing).astype(int)
    
    # 确保所有点都在网格范围内
    grid_indices[:, 0] = np.clip(grid_indices[:, 0], 0, grid_size_x - 1)
    grid_indices[:, 1] = np.clip(grid_indices[:, 1], 0, grid_size_y - 1)
    
    # 统计每个网格内的点数
    for idx in grid_indices:
        grid[idx[1], idx[0]] += 1
    
    return grid
def save_grid_as_csv(grid, filename):
    """保存为CSV文件"""
    np.savetxt(filename, grid, delimiter=',', fmt='%d')
    print(f"网格已保存为CSV文件：{filename}")
def save_grid_as_txt(grid, filename):
    """保存为文本文件"""
    np.savetxt(filename, grid, fmt='%d')
    print(f"网格已保存为文本文件：{filename}")
def save_txt(array, filename):
    """保存为文本文件"""
    np.savetxt(filename, array, fmt='%.4f')
    print(f"数组已保存为文本文件：{filename}")
def calculate_correlation_matrix(input_file: str, output_file: str, delimiter: str = None) -> np.ndarray:
    """
    从txt文件读取数据，计算相关系数矩阵并保存到另一个txt文件
    
    Parameters:
    -----------
    input_file : str
        输入txt文件路径
    output_file : str 
        输出txt文件路径
    delimiter : str, optional
        数据分隔符，默认为None（按空格分割）
    
    Returns:
    --------
    np.ndarray
        相关系数矩阵
    """
    
    try:
        # 读取数据
        print(f"正在读取文件: {input_file}")
        if delimiter:
            data = np.loadtxt(input_file, delimiter=delimiter)
        else:
            data = np.loadtxt(input_file)
        
        # 检查数据维度
        if len(data.shape) == 1:
            # 如果是一维数据，转换为二维
            data = data.reshape(-1, 1)
            print("检测到一维数据，已转换为二维数组")
        
        n_samples, n_features = data.shape
        print(f"数据读取成功: {n_samples} 个样本, {n_features} 个特征")
        
        # 计算相关系数矩阵
        print("正在计算相关系数矩阵...")
        correlation_matrix = np.corrcoef(data)
        
        # 转换矩阵元素：1/(相关系数的绝对值+0.0001) - (1/1.0001)
        print("正在转换矩阵元素...")
        transformed_matrix = 1 / (np.abs(correlation_matrix) + 0.0001) - (1 / 1.0001)
        
        # 保存转换后的矩阵到文件
        print(f"正在保存结果到: {output_file}")
        np.savetxt(output_file, transformed_matrix, fmt='%.6f', delimiter='\t')
        
        print("相关系数矩阵计算完成!")
        return correlation_matrix
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        raise
    except Exception as e:
        print(f"错误: {str(e)}")
        raise
matrix=np.loadtxt('聚类分析作业\\mid_result\\SP500_1_matrix.txt')
# Global variables - to be set by the user before using the custom distance function
# data_matrix should contain all sequences (2D array where each row is a sequence)
# dist_matrix should be the precomputed distance matrix (2D array where dist_matrix[i,j] is distance between sequence i and j)
data_matrix = None
dist_matrix = None

def my_distance(x, y):
    """
    Custom distance function for sklearn.cluster algorithms.
    This function uses precomputed distance matrix to return the distance between two sequences.
    
    Parameters:
    x, y : array-like
        Two sequences (1D arrays) from the data_matrix
        
    Returns:
    float
        The precomputed distance between x and y from dist_matrix
        
    Raises:
    ValueError if data_matrix or dist_matrix are not set, or if vectors are not found in data_matrix
    """
    if data_matrix is None or dist_matrix is None:
        raise ValueError("data_matrix and dist_matrix must be set before using this function")
    
    # Find the indices of x and y in data_matrix
    i = find_index(data_matrix, x)
    j = find_index(data_matrix, y)
    return dist_matrix[i, j]

def find_index(data, vec):
    """
    Find the index of a vector in the data array using floating point comparison with tolerance.
    
    Parameters:
    data : 2D array
        Array containing all sequences
    vec : 1D array
        Vector to find in data
        
    Returns:
    int
        Index of the vector in data
        
    Raises:
    ValueError if vector is not found in data
    """
    if len(data) == 0:
        raise ValueError("data_matrix is empty")
    
    # Use numpy isclose for floating point comparison with default tolerance
    matches = np.all(np.isclose(data, vec), axis=1)
    indices = np.where(matches)[0]
    
    if len(indices) == 0:
        raise ValueError("Vector not found in data_matrix")
    
    return indices[0]

# Alternative function that takes indices directly (useful for testing)
def sequence_distance(i, j):
    """
    Direct access to precomputed distance matrix using indices.
    This can be used when you know the indices of the sequences.
    
    Parameters:
    i, j : int
        Indices of the sequences in data_matrix
        
    Returns:
    float
        The distance between sequence i and sequence j
    """
    if dist_matrix is None:
        raise ValueError("dist_matrix must be set before using this function")
    return dist_matrix[i, j]

def visualize_clusters(X, labels):
    """
    可视化序列数据及其聚类标签
    
    Parameters:
    -----------
    X : np.ndarray
        形状为 (n_samples, n_features) 的序列数据
    labels : np.ndarray
        形状为 (n_samples,) 的聚类标签
    """
    # 获取唯一的聚类标签和簇的数量
    unique_labels = np.unique(labels)
    n_clusters_found = len(unique_labels)
    
    # 为每个簇分配一个颜色
    # matplotlib 的 'tab10' 颜色映射提供了10种易于区分的颜色
    colors = plt.cm.get_cmap('tab10', n_clusters_found)
    
    # 创建图形和坐标轴
    plt.figure(figsize=(12, 8))
    
    # 遍历每个簇
    for i, cluster_id in enumerate(unique_labels):
        # 筛选出属于当前簇的所有数据点
        cluster_data = X[labels == cluster_id]
        
        # 遍历当前簇的每一条序列
        for sequence in cluster_data:
            # 绘制折线，使用当前簇对应的颜色
            plt.plot(sequence, color=colors(i), alpha=0.5) # alpha设置透明度，使重叠部分可见
            
    # 添加图例和标题
    plt.title('Visualization of Clustered Time Series Data')
    plt.xlabel('Time Step / Feature Index')
    plt.ylabel('Value')
    # 创建图例，每个标签对应一个颜色块
    handles = [plt.Line2D([0], [0], color=colors(i), lw=4) for i in range(n_clusters_found)]
    plt.legend(handles, [f'Cluster {i}' for i in unique_labels])
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()   
'''
dataset=pd.read_csv('聚类分析作业\\dataset\\CC GENERAL.csv')
dataset=dataset.drop(['CUST_ID'],axis=1)
dataset=dataset.dropna()
X=pd.DataFrame(StandardScaler().fit_transform(dataset))
X=np.asarray(X)
pca=PCA(n_components=2,random_state=24)
X_pca=pca.fit_transform(X)
save_txt(X_pca,'聚类分析作业\\mid_result\\GENERAL.txt')
'''
'''
SP500 = np.genfromtxt('chapter8\\data\\SP500array.csv', delimiter=',').T
nStock = len(SP500[:, 0])

first_column = SP500[:, [0]]
normalized_matrix=SP500/first_column
save_txt(normalized_matrix,'聚类分析作业\\mid_result\\SP500_1.txt')

diffs = np.diff(SP500, axis=1)
denominator = SP500[:, :-1]
returns = diffs / denominator
normalized_matrix = np.c_[np.zeros(SP500.shape[0]), returns]
save_txt(normalized_matrix,'聚类分析作业\\mid_result\\SP500_2.txt')
'''
'''
calculate_correlation_matrix(input_file='聚类分析作业\\mid_result\\SP500_1.txt',output_file='聚类分析作业\\mid_result\\SP500_1_matrix.txt')
calculate_correlation_matrix(input_file='聚类分析作业\\mid_result\\SP500_2.txt',output_file='聚类分析作业\\mid_result\\SP500_2_matrix.txt')
'''