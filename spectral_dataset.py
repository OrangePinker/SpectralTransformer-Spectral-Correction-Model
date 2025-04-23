import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SpectralDataset(Dataset):
    def __init__(self, data, labels):
        """
        初始化光谱数据集
        
        参数:
            data: 光谱数据，形状为 [n_samples, n_features]
            labels: 标签，形状为 [n_samples]
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def plot_first_column_data(folder_path, background_path, output_dir):
    """
    绘制每个Excel文件第一列的原始数据和预处理后的数据，分别输出为两张图片
    
    参数:
        folder_path: 包含Excel文件的文件夹路径
        background_path: 背景数据Excel文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取背景数据
    print(f"读取背景数据: {background_path}")
    bg_df = pd.read_excel(background_path, header=None)
    bg_labels = bg_df.iloc[0, :].values
    background_data = bg_df.iloc[1:, :].values
    
    # 用于存储所有类别的颜色
    colors = plt.cm.tab10.colors
    
    # 创建原始数据图表
    plt.figure(figsize=(12, 8))
    plt.title('Original Spectral Data (First Column of Each Excel)')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Intensity')
    plt.grid(True)
    
    # 存储预处理后的数据，稍后绘制
    processed_data_dict = {}
    
    # 遍历文件夹中的所有Excel文件
    for idx, filename in enumerate(os.listdir(folder_path)):
        if (filename.endswith('.xlsx') or filename.endswith('.xls')) and not filename.startswith('~$'):
            file_path = os.path.join(folder_path, filename)
            class_name = os.path.splitext(filename)[0]
            
            # 读取Excel文件
            df = pd.read_excel(file_path, header=None)
            column_labels = df.iloc[0, :].values
            spectral_data = df.iloc[1:, :].values
            
            # 获取第一列数据
            if spectral_data.shape[1] > 0:
                first_col_data = spectral_data[:, 0]
                
                # 绘制原始数据
                plt.plot(first_col_data, color=colors[idx % len(colors)], label=class_name)
                
                # 预处理第一列数据
                col_label = column_labels[0]
                col_label_str = str(col_label).strip()
                bg_labels_str = [str(label).strip() for label in bg_labels]
                
                # 查找匹配的背景数据
                if col_label_str in bg_labels_str:
                    bg_col_idx = bg_labels_str.index(col_label_str)
                else:
                    # 尝试数值匹配
                    try:
                        col_label_float = float(col_label_str)
                        bg_labels_float = [float(str(label).strip()) if str(label).strip().replace('.', '', 1).isdigit() else float('inf') for label in bg_labels]
                        closest_idx = np.argmin(np.abs(np.array(bg_labels_float) - col_label_float))
                        if abs(bg_labels_float[closest_idx] - col_label_float) < 0.001:
                            bg_col_idx = closest_idx
                        else:
                            print(f"警告: 在背景数据中找不到标签 {col_label}，跳过此样本")
                            continue
                    except (ValueError, TypeError):
                        print(f"警告: 在背景数据中找不到标签 {col_label}，跳过此样本")
                        continue
                
                # 确保数据长度匹配
                bg_sample = background_data[:, bg_col_idx]
                if first_col_data.shape[0] != bg_sample.shape[0]:
                    min_len = min(first_col_data.shape[0], bg_sample.shape[0])
                    first_col_data = first_col_data[:min_len]
                    bg_sample = bg_sample[:min_len]
                
                # 预处理：除以背景数据并归一化
                bg_sample = np.where(bg_sample == 0, 1e-10, bg_sample)
                processed_data = first_col_data / bg_sample
                
                # 线性归一化到[0,1]区间
                min_val = np.min(processed_data)
                max_val = np.max(processed_data)
                if max_val > min_val:
                    normalized_data = (processed_data - min_val) / (max_val - min_val)
                else:
                    normalized_data = np.zeros_like(processed_data)
                
                # 存储预处理后的数据
                processed_data_dict[class_name] = normalized_data
    
    # 添加图例并保存原始数据图表
    plt.legend(loc='upper right')
    original_output_path = os.path.join(output_dir, 'original_spectral_data.png')
    plt.savefig(original_output_path, dpi=300, bbox_inches='tight')
    print(f"原始光谱数据图已保存到: {original_output_path}")
    plt.close()
    
    # 创建预处理后的数据图表
    plt.figure(figsize=(12, 8))
    plt.title('Preprocessed Spectral Data (First Column of Each Excel)')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Normalized Intensity')
    plt.grid(True)
    
    # 绘制预处理后的数据
    for idx, (class_name, data) in enumerate(processed_data_dict.items()):
        plt.plot(data, color=colors[idx % len(colors)], label=class_name)
    
    # 添加图例并保存预处理后的数据图表
    plt.legend(loc='upper right')
    processed_output_path = os.path.join(output_dir, 'preprocessed_spectral_data.png')
    plt.savefig(processed_output_path, dpi=300, bbox_inches='tight')
    print(f"预处理后的光谱数据图已保存到: {processed_output_path}")
    plt.close()

def load_spectral_data(folder_path, background_path, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    从文件夹中加载所有Excel文件的光谱数据，并划分为训练集、验证集和测试集
    每个Excel文件读取后都会随机打乱数据
    
    参数:
        folder_path: 包含Excel文件的文件夹路径
        background_path: 背景数据Excel文件路径
        train_size: 训练集比例
        val_size: 验证集比例
        test_size: 测试集比例
        random_state: 随机种子
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        class_names: 类别名称列表
    """
    # 读取背景数据
    print(f"读取背景数据: {background_path}")
    bg_df = pd.read_excel(background_path, header=None)  # 不使用标题行
    
    # 获取背景数据的标签（第一行）
    bg_labels = bg_df.iloc[0, :].values
    # 获取背景数据（第2行到最后一行）
    background_data = bg_df.iloc[1:, :].values
    
    # 存储所有处理后的数据和标签
    all_data = []
    all_labels = []
    class_names = []
    
    # 遍历文件夹中的所有Excel文件
    for class_idx, filename in enumerate(os.listdir(folder_path)):
        if (filename.endswith('.xlsx') or filename.endswith('.xls')) and not filename.startswith('~$'):
            file_path = os.path.join(folder_path, filename)
            class_name = os.path.splitext(filename)[0]
            class_names.append(class_name)
            
            print(f"处理文件: {filename}, 类别: {class_name}")
            
            # 读取Excel文件
            df = pd.read_excel(file_path, header=None)
            column_labels = df.iloc[0, :].values
            spectral_data = df.iloc[1:, :].values
            
            # 存储当前文件的处理后数据
            file_processed_data = []
            
            # 处理每一列数据
            for col_idx in range(spectral_data.shape[1]):
                col_data = spectral_data[:, col_idx]
                col_label = column_labels[col_idx]
                
                # 将列标签转换为字符串并去除空格
                col_label_str = str(col_label).strip()
                bg_labels_str = [str(label).strip() for label in bg_labels]
                
                # 查找匹配的背景数据
                if col_label_str in bg_labels_str:
                    bg_col_idx = bg_labels_str.index(col_label_str)
                else:
                    # 尝试数值匹配
                    try:
                        col_label_float = float(col_label_str)
                        bg_labels_float = [float(str(label).strip()) if str(label).strip().replace('.', '', 1).isdigit() else float('inf') for label in bg_labels]
                        closest_idx = np.argmin(np.abs(np.array(bg_labels_float) - col_label_float))
                        if abs(bg_labels_float[closest_idx] - col_label_float) < 0.001:
                            bg_col_idx = closest_idx
                        else:
                            print(f"警告: 在背景数据中找不到标签 {col_label}，跳过此样本")
                            continue
                    except (ValueError, TypeError):
                        print(f"警告: 在背景数据中找不到标签 {col_label}，跳过此样本")
                        continue
                
                # 确保数据长度匹配
                bg_sample = background_data[:, bg_col_idx]
                if col_data.shape[0] != bg_sample.shape[0]:
                    min_len = min(col_data.shape[0], bg_sample.shape[0])
                    col_data = col_data[:min_len]
                    bg_sample = bg_sample[:min_len]
                
                # 恢复预处理步骤：背景去除和归一化
                bg_sample = np.where(bg_sample == 0, 1e-10, bg_sample)  # 避免除以零
                processed_data = col_data / bg_sample
                
                # 恢复归一化步骤
                min_val = np.min(processed_data)
                max_val = np.max(processed_data)
                if max_val > min_val:
                    normalized_data = (processed_data - min_val) / (max_val - min_val)
                else:
                    normalized_data = np.zeros_like(processed_data)
                
                # 使用归一化后的数据
                file_processed_data.append(normalized_data)
            
            # 将处理后的数据转换为numpy数组
            if file_processed_data:
                file_processed_data = np.array(file_processed_data)
                
                # 随机打乱当前Excel文件的数据
                n_samples = file_processed_data.shape[0]
                shuffle_indices = np.random.permutation(n_samples)
                file_processed_data = file_processed_data[shuffle_indices]
                
                # 添加到总数据集
                all_data.append(file_processed_data)
                all_labels.extend([class_idx] * n_samples)
                print(f"  添加了 {n_samples} 个样本，类别 {class_name}")
    
    # 将所有数据合并
    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    # 再次随机打乱所有数据
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    
    print(f"总共加载了 {X.shape[0]} 个样本，{len(class_names)} 个类别")
    
    # 划分训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 进一步划分训练集和验证集
    train_ratio = train_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, train_size=train_ratio, 
        random_state=random_state, stratify=y_train_val
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建数据集
    train_dataset = SpectralDataset(X_train, y_train)
    val_dataset = SpectralDataset(X_val, y_val)
    test_dataset = SpectralDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_names