import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def parse_clustering_metrics(file_path):
    """
    解析聚类评估指标文件
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 使用正则表达式提取数据
    pattern = r'([\w_]+)聚类评估指标:\s*轮廓系数:\s*([\d.]+)\s*Calinski-Harabasz指数:\s*([\d.]+)\s*Davies-Bouldin指数:\s*([\d.]+)'
    matches = re.findall(pattern, content)
    
    data = []
    for match in matches:
        algorithm = match[0]
        silhouette = float(match[1])
        calinski_harabasz = float(match[2])
        davies_bouldin = float(match[3])
        
        # 提取聚类数目
        cluster_num_match = re.search(r'_(\d+)$', algorithm)
        if cluster_num_match:
            cluster_num = int(cluster_num_match.group(1))
        else:
            # 对于没有明确数字的算法，尝试从名称中提取
            if 'dbscan' in algorithm:
                cluster_num = 0  # DBSCAN自动确定聚类数
            elif 'OPTICS' in algorithm:
                cluster_num = 0
            else:
                cluster_num = None
        
        data.append({
            'algorithm': algorithm,
            'cluster_num': cluster_num,
            'silhouette_score': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin
        })
    
    return pd.DataFrame(data)

def create_visualizations(df):
    """
    创建聚类指标的可视化图表（不使用seaborn）
    """
    # 设置图形样式
    plt.figure(figsize=(18, 12))
    
    # 1. 轮廓系数比较
    plt.subplot(2, 2, 1)
    algorithms = df['algorithm'].tolist()
    silhouette_scores = df['silhouette_score'].tolist()
    
    bars = plt.bar(range(len(algorithms)), silhouette_scores, color='skyblue', alpha=0.7)
    plt.title('各聚类算法的轮廓系数比较', fontsize=14, fontweight='bold')
    plt.xlabel('算法')
    plt.ylabel('轮廓系数')
    plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
    
    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 2. Calinski-Harabasz指数比较
    plt.subplot(2, 2, 2)
    calinski_scores = df['calinski_harabasz'].tolist()
    
    bars = plt.bar(range(len(algorithms)), calinski_scores, color='lightgreen', alpha=0.7)
    plt.title('各聚类算法的Calinski-Harabasz指数比较', fontsize=14, fontweight='bold')
    plt.xlabel('算法')
    plt.ylabel('Calinski-Harabasz指数')
    plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
    
    # 3. Davies-Bouldin指数比较
    plt.subplot(2, 2, 3)
    davies_scores = df['davies_bouldin'].tolist()
    
    bars = plt.bar(range(len(algorithms)), davies_scores, color='lightcoral', alpha=0.7)
    plt.title('各聚类算法的Davies-Bouldin指数比较', fontsize=14, fontweight='bold')
    plt.xlabel('算法')
    plt.ylabel('Davies-Bouldin指数')
    plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
    
    # 4. 综合性能雷达图
    plt.subplot(2, 2, 4)
    
    # 标准化数据
    normalized_silhouette = (df['silhouette_score'] - df['silhouette_score'].min()) / (df['silhouette_score'].max() - df['silhouette_score'].min())
    normalized_calinski = (df['calinski_harabasz'] - df['calinski_harabasz'].min()) / (df['calinski_harabasz'].max() - df['calinski_harabasz'].min())
    # Davies-Bouldin指数越低越好，所以取1-标准化值
    normalized_davies = 1 - (df['davies_bouldin'] - df['davies_bouldin'].min()) / (df['davies_bouldin'].max() - df['davies_bouldin'].min())
    
    # 选择前6个算法显示在雷达图上
    num_algorithms = min(6, len(df))
    
    # 雷达图角度
    angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 创建雷达图
    for i in range(num_algorithms):
        values = [
            normalized_silhouette.iloc[i],
            normalized_calinski.iloc[i], 
            normalized_davies.iloc[i]
        ]
        values += values[:1]  # 闭合图形
        
        plt.polar(angles, values, 'o-', linewidth=2, label=df['algorithm'].iloc[i])
        plt.fill(angles, values, alpha=0.1)
    
    # 设置雷达图标签
    plt.xticks(angles[:-1], ['轮廓系数', 'Calinski-Harabasz', 'Davies-Bouldin'])
    plt.title('算法性能雷达图（标准化）', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('clustering_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 性能对比表格
    print("\n=== 聚类算法性能对比 ===")
    print("算法\t\t\t轮廓系数\tCalinski-Harabasz\tDavies-Bouldin")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['algorithm']:20}\t{row['silhouette_score']:.4f}\t\t{row['calinski_harabasz']:.1f}\t\t\t{row['davies_bouldin']:.4f}")

def main():
    """
    主函数
    """
    file_path = '聚类分析作业\\result\\1000points\\result_points.txt'
    
    print("正在解析聚类评估指标文件...")
    df = parse_clustering_metrics(file_path)
    
    print(f"成功解析 {len(df)} 个算法的评估指标")
    print("\n数据概览:")
    print(df.head())
    
    print("\n创建可视化图表...")
    create_visualizations(df)
    
    print("可视化完成！生成的图表已保存到当前目录。")
    
    # 显示最佳性能算法
    print("\n=== 最佳性能算法 ===")
    best_silhouette = df.loc[df['silhouette_score'].idxmax()]
    best_calinski = df.loc[df['calinski_harabasz'].idxmax()]
    best_davies = df.loc[df['davies_bouldin'].idxmin()]  # Davies-Bouldin越低越好
    
    print(f"最佳轮廓系数: {best_silhouette['algorithm']} - {best_silhouette['silhouette_score']:.4f}")
    print(f"最佳Calinski-Harabasz指数: {best_calinski['algorithm']} - {best_calinski['calinski_harabasz']:.4f}")
    print(f"最佳Davies-Bouldin指数: {best_davies['algorithm']} - {best_davies['davies_bouldin']:.4f}")

if __name__ == "__main__":
    main()
