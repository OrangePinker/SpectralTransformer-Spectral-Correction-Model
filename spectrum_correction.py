import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义数据集类
class SpectralDataset(Dataset):
    def __init__(self, distorted_spectra, normal_spectra, currents, material_ids=None):
        """
        初始化光谱数据集
        
        参数:
            distorted_spectra: 畸变光谱数据 [样本数, 通道数]
            normal_spectra: 正常光谱数据 [样本数, 通道数]
            currents: 管电流值 [样本数]
            material_ids: 物质ID [样本数]
        """
        self.distorted_spectra = torch.tensor(distorted_spectra, dtype=torch.float32)
        self.normal_spectra = torch.tensor(normal_spectra, dtype=torch.float32)
        self.currents = torch.tensor(currents, dtype=torch.float32).unsqueeze(1)
        
        # 添加物质ID
        if material_ids is not None:
            self.material_ids = torch.tensor(material_ids, dtype=torch.long)
        else:
            self.material_ids = torch.zeros(len(distorted_spectra), dtype=torch.long)
        
    def __len__(self):
        return len(self.distorted_spectra)
    
    def __getitem__(self, idx):
        return {
            'distorted_spectrum': self.distorted_spectra[idx],
            'current': self.currents[idx],
            'normal_spectrum': self.normal_spectra[idx],
            'material_id': self.material_ids[idx]
        }

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# 定义Transformer模型
class SpectralTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1, num_materials=1):
        super(SpectralTransformer, self).__init__()
        
        # 输入投影层 - 修改为接受3维输入（光谱值、电流值和物质ID）
        self.input_projection = nn.Linear(3, d_model)  # 3 = 1(光谱值) + 1(电流值) + 1(物质ID嵌入)
        
        # 物质嵌入层
        self.material_embedding = nn.Embedding(num_materials, 1)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器和解码器
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, 1)  # 输出单个值
        
        # 电流嵌入层
        self.current_embedding = nn.Linear(1, d_model)
    
    def forward(self, src, current, material_id=None, tgt=None):
        # 将电流值与每个光谱点连接
        batch_size, seq_len = src.shape
        
        # 修改这里：正确处理维度
        # 将current从[batch_size, 1]扩展为[batch_size, seq_len, 1]
        current_expanded = current.repeat(1, seq_len).unsqueeze(-1)  # [batch_size, seq_len, 1]
        src_expanded = src.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # 添加物质ID嵌入
        if material_id is None:
            material_id = torch.zeros(batch_size, dtype=torch.long, device=src.device)
        
        # 获取物质嵌入并扩展到与光谱相同的序列长度
        material_emb = self.material_embedding(material_id)  # [batch_size, 1]
        material_expanded = material_emb.repeat(1, seq_len).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # 连接src、current和material_id，得到[batch_size, seq_len, 3]
        src_with_features = torch.cat([src_expanded, current_expanded, material_expanded], dim=2)
        
        # 输入投影
        src = self.input_projection(src_with_features)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        src = self.pos_encoder(src)
        
        # 如果没有提供目标序列（推理阶段），则使用电流值创建一个序列
        if tgt is None:
            tgt = self.current_embedding(current).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            # 将目标序列与电流值和物质ID连接并投影
            tgt_expanded = tgt.unsqueeze(-1)  # [batch_size, seq_len, 1]
            tgt_with_features = torch.cat([tgt_expanded, current_expanded, material_expanded], dim=2)
            tgt = self.input_projection(tgt_with_features)
            tgt = self.pos_encoder(tgt)
        
        # 创建掩码（解码器中使用）
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # 通过Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = self.output_projection(output)
        
        # 添加残差连接 - 直接将输入添加到输出
        if tgt is None:  # 推理模式
            output = output.squeeze(-1) + src_expanded.squeeze(-1) * 0.3  # 添加一个缩放的残差连接
        else:
            output = output.squeeze(-1)  # 训练模式保持不变
        
        return output  # 确保输出形状与输入相同
# 数据加载和预处理函数
def load_and_preprocess_data(normal_folder_path, distorted_folder_path):
    """
    从文件夹中加载多个Excel文件并预处理数据
    
    参数:
        normal_folder_path: 正常光谱数据文件夹路径
        distorted_folder_path: 畸变光谱数据文件夹路径
    
    返回:
        处理后的数据集
    """
    import os
    
    # 获取文件夹中的所有Excel文件
    normal_files = [f for f in os.listdir(normal_folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]
    distorted_files = [f for f in os.listdir(distorted_folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]
    
    # 找出两个文件夹中共有的文件名
    common_files = set([f for f in normal_files if f in distorted_files])
    
    if len(common_files) == 0:
        raise ValueError("没有找到两个文件夹中具有相同名称的Excel文件！")
    
    print(f"找到 {len(common_files)} 对匹配的Excel文件")
    
    # 初始化数据列表
    all_normal_spectra = []
    all_distorted_spectra = []
    all_currents = []
    all_material_ids = []  # 新增：记录每个样本对应的物质ID
    
    # 处理每对文件
    for material_id, file_name in enumerate(common_files):
        normal_file_path = os.path.join(normal_folder_path, file_name)
        distorted_file_path = os.path.join(distorted_folder_path, file_name)
        
        print(f"处理文件对: {file_name} (物质ID: {material_id})")
        
        # 读取Excel文件
        normal_df = pd.read_excel(normal_file_path, header=None)
        distorted_df = pd.read_excel(distorted_file_path, header=None)
        
        # 提取管电流值（第一行）
        currents = normal_df.iloc[0, :].values
        
        # 提取光谱数据（第2行到第513行）
        normal_spectra = normal_df.iloc[1:513, :].values.T  # 转置使得每行是一个样本
        distorted_spectra = distorted_df.iloc[1:513, :].values.T
        
        # 确保数据维度一致
        print(f"  原始数据维度 - 正常光谱: {normal_spectra.shape}, 畸变光谱: {distorted_spectra.shape}, 电流值: {currents.shape}")
        
        # 确保所有数据的样本数量一致
        min_samples = min(normal_spectra.shape[0], distorted_spectra.shape[0], currents.shape[0])
        normal_spectra = normal_spectra[:min_samples]
        distorted_spectra = distorted_spectra[:min_samples]
        currents = currents[:min_samples]
        
        # 为每个样本添加物质ID
        material_ids = np.full(min_samples, material_id)
        
        # 添加到总数据集
        all_normal_spectra.append(normal_spectra)
        all_distorted_spectra.append(distorted_spectra)
        all_currents.append(currents)
        all_material_ids.append(material_ids)
    
    # 合并所有数据
    normal_spectra_combined = np.vstack(all_normal_spectra)
    distorted_spectra_combined = np.vstack(all_distorted_spectra)
    currents_combined = np.concatenate(all_currents)
    material_ids_combined = np.concatenate(all_material_ids)
    
    print(f"合并后数据维度 - 正常光谱: {normal_spectra_combined.shape}, 畸变光谱: {distorted_spectra_combined.shape}, 电流值: {currents_combined.shape}, 物质ID: {material_ids_combined.shape}")
    
    # 计算并打印每个样本的总强度，用于验证
    normal_sums = np.sum(normal_spectra_combined, axis=1)
    distorted_sums = np.sum(distorted_spectra_combined, axis=1)
    print(f"正常光谱平均总强度: {np.mean(normal_sums):.2f}, 畸变光谱平均总强度: {np.mean(distorted_sums):.2f}")
    print(f"正常/畸变光谱总强度比例: {np.mean(normal_sums/distorted_sums):.4f}")
    
    # 保存原始数据的总强度比例，用于后续校正
    intensity_ratios = normal_sums / distorted_sums
    
    # 数据归一化 - 将范围限制在0-0.95之间
    spectra_scaler = MinMaxScaler(feature_range=(0, 0.95))
    normal_spectra_scaled = spectra_scaler.fit_transform(normal_spectra_combined)
    distorted_spectra_scaled = spectra_scaler.transform(distorted_spectra_combined)
    
    current_scaler = MinMaxScaler()
    currents_scaled = current_scaler.fit_transform(currents_combined.reshape(-1, 1)).flatten()
    
    # 保存物质名称映射
    material_names = list(common_files)
    material_map = {i: name for i, name in enumerate(material_names)}
    
    return normal_spectra_scaled, distorted_spectra_scaled, currents_scaled, material_ids_combined, spectra_scaler, current_scaler, intensity_ratios, material_map

# 自定义损失函数，同时考虑MSE和总强度差异
class SpectralLoss(nn.Module):
    def __init__(self, intensity_weight=1.5, peak_weight=0.5, linearity_weight=1.0):
        super(SpectralLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.intensity_weight = intensity_weight
        self.peak_weight = peak_weight
        self.linearity_weight = linearity_weight
    
    def forward(self, output, target, current=None):
        # 计算MSE损失
        mse_loss = self.mse(output, target)
        
        # 计算总强度损失
        output_sum = torch.sum(output, dim=1)
        target_sum = torch.sum(target, dim=1)
        intensity_loss = torch.mean(torch.abs(output_sum - target_sum) / target_sum)
        
        # 添加峰值约束
        output_max = torch.max(output, dim=1)[0]
        target_max = torch.max(target, dim=1)[0]
        peak_loss = torch.mean(torch.abs(output_max - target_max) / target_max)
        
        # 添加电流-光子数线性关系约束
        linearity_loss = 0.0
        if current is not None:
            # 计算当前批次中电流值与总光子数的相关性
            if len(current) > 1:  # 至少需要两个样本才能计算相关性
                current_np = current.cpu().numpy().flatten()
                output_sum_np = output_sum.detach().cpu().numpy()
                
                # 计算皮尔逊相关系数
                from scipy.stats import pearsonr
                try:
                    corr, _ = pearsonr(current_np, output_sum_np)
                    # 我们希望相关系数接近1（完美正相关）
                    linearity_loss = 1.0 - corr
                except:
                    linearity_loss = 0.0  # 如果计算失败，不添加这个损失
        
        # 组合损失
        total_loss = mse_loss + self.intensity_weight * intensity_loss + self.peak_weight * peak_loss
        if current is not None:
            total_loss += self.linearity_weight * linearity_loss
        
        return total_loss
# 光谱校正函数
def correct_spectrum(model, distorted_spectrum, current, material_id, spectra_scaler, current_scaler, 
                    intensity_ratio=None, normal_spectra=None, currents=None, material_ids=None):
    """
    使用训练好的模型校正畸变光谱
    
    参数:
        model: 训练好的模型
        distorted_spectrum: 畸变光谱 [batch_size, seq_len]
        current: 管电流值 [batch_size, 1]
        material_id: 物质ID [batch_size]
        spectra_scaler: 光谱归一化器
        current_scaler: 电流归一化器
        intensity_ratio: 强度比例，用于校正总强度
        normal_spectra: 正常光谱数据，用于参考总光子数 [样本数, 通道数]
        currents: 所有样本的电流值，用于建立电流-光子数关系 [样本数]
        material_ids: 所有样本的物质ID [样本数]
    
    返回:
        校正后的光谱 [batch_size, seq_len]
    """
    model.eval()
    with torch.no_grad():
        # 前向传播获取校正后的光谱
        corrected_spectrum_scaled = model(distorted_spectrum, current, material_id)
        
        # 转换为CPU并转换为NumPy数组
        corrected_spectrum_scaled = corrected_spectrum_scaled.cpu()
        
        # 反归一化输出数据
        corrected_spectrum = spectra_scaler.inverse_transform(corrected_spectrum_scaled.numpy())
        
        # 确保输出非负
        corrected_spectrum = np.clip(corrected_spectrum, 0, None)
        
        # 应用更精确的强度校正，按物质ID分组
        if normal_spectra is not None and currents is not None and material_ids is not None:
            # 获取当前批次的物质ID
            batch_material_ids = material_id.cpu().numpy()
            
            for i in range(len(corrected_spectrum)):
                current_material_id = batch_material_ids[i]
                
                # 找出训练集中相同物质ID的样本
                same_material_mask = (material_ids == current_material_id)
                
                if np.sum(same_material_mask) > 0:
                    # 反归一化该物质的正常光谱数据
                    material_normal_spectra = spectra_scaler.inverse_transform(normal_spectra[same_material_mask])
                    material_currents = currents[same_material_mask]
                    
                    # 计算该物质正常光谱的总光子数
                    material_normal_sums = np.sum(material_normal_spectra, axis=1)
                    
                    # 建立该物质的电流值与总光子数的线性关系
                    from sklearn.linear_model import LinearRegression
                    material_photon_model = LinearRegression()
                    material_photon_model.fit(material_currents.reshape(-1, 1), material_normal_sums)
                    
                    # 获取当前电流值对应的预期总光子数
                    current_np = current.cpu().numpy()[i:i+1]
                    expected_sum = material_photon_model.predict(current_np)
                    
                    # 应用校正
                    actual_sum = np.sum(corrected_spectrum[i])
                    if actual_sum > 0:  # 避免除零错误
                        scale_factor = expected_sum[0] / actual_sum
                        # 限制缩放因子范围，避免过度校正
                        scale_factor = np.clip(scale_factor, 0.8, 1.2)
                        corrected_spectrum[i] = corrected_spectrum[i] * scale_factor
        
        # 如果没有正常光谱数据，则使用传统的强度比例校正
        elif intensity_ratio is not None:
            # 获取原始畸变光谱（反归一化）
            distorted_spectrum_np = spectra_scaler.inverse_transform(distorted_spectrum.cpu().numpy())
            
            # 计算当前总强度
            distorted_sum = np.sum(distorted_spectrum_np, axis=1)
            corrected_sum = np.sum(corrected_spectrum, axis=1)
            
            # 应用校正比例
            for i in range(len(distorted_sum)):
                target_sum = distorted_sum[i] * intensity_ratio
                if corrected_sum[i] > 0:  # 避免除零错误
                    scale_factor = target_sum / corrected_sum[i]
                    scale_factor = min(scale_factor, 1.2)  # 限制最大缩放比例
                    corrected_spectrum[i] = corrected_spectrum[i] * scale_factor
        
        # 将结果转换为整数
        corrected_spectrum = np.round(corrected_spectrum).astype(int)
        
        return torch.tensor(corrected_spectrum, dtype=torch.float32)

# 修改可视化函数以支持物质ID
def visualize_results(model, test_loader, device, spectra_scaler, current_scaler, 
                     normal_spectra=None, currents=None, material_ids=None, 
                     intensity_ratios=None, num_samples=3, material_map=None):
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            distorted_spectrum = batch['distorted_spectrum'].to(device)
            current = batch['current'].to(device)
            normal_spectrum = batch['normal_spectrum'].to(device)
            material_id = batch['material_id'].to(device)
            
            # 获取物质名称
            material_name = material_map[material_id.cpu().numpy()[0]] if material_map else f"物质{material_id.cpu().numpy()[0]}"
            
            # 使用更新后的correct_spectrum函数进行校正
            corrected_spectrum = correct_spectrum(
                model, 
                distorted_spectrum, 
                current,
                material_id,
                spectra_scaler, 
                current_scaler,
                intensity_ratio=np.mean(intensity_ratios) if intensity_ratios is not None else None,
                normal_spectra=normal_spectra if normal_spectra is not None else None,
                currents=currents if currents is not None else None,
                material_ids=material_ids if material_ids is not None else None
            )
            
            # 转换为CPU并转换为NumPy数组
            distorted_spectrum_np = distorted_spectrum.cpu().numpy()
            normal_spectrum_np = normal_spectrum.cpu().numpy()
            corrected_spectrum_np = corrected_spectrum.cpu().numpy()
            
            # 计算并打印总光子数，用于验证校正效果
            distorted_sum = np.sum(spectra_scaler.inverse_transform(distorted_spectrum_np)[0])
            normal_sum = np.sum(spectra_scaler.inverse_transform(normal_spectrum_np)[0])
            corrected_sum = np.sum(corrected_spectrum_np[0])
            current_val = current.cpu().numpy()[0][0]
            
            print(f"样本 {i+1} - 物质: {material_name} - 电流值: {current_val:.4f}")
            print(f"  畸变光谱总光子数: {distorted_sum:.0f}")
            print(f"  正常光谱总光子数: {normal_sum:.0f}")
            print(f"  校正后光谱总光子数: {corrected_sum:.0f}")
            print(f"  校正/正常比例: {corrected_sum/normal_sum:.4f}")
            
            # 绘制结果
            plt.figure(figsize=(12, 6))
            plt.plot(spectra_scaler.inverse_transform(distorted_spectrum_np)[0], 
                    label='Distorted spectrum', alpha=0.7)
            plt.plot(spectra_scaler.inverse_transform(normal_spectrum_np)[0], 
                    label='Normal spectrum', alpha=0.7)
            plt.plot(corrected_spectrum_np[0], 
                    label='Corrected spectrum', alpha=0.7)
            plt.title(f'Sample {i+1} - Material: {material_name} - Current: {current_val:.4f}')
            plt.xlabel('Channels[1-512]')
            plt.ylabel('Photon Counts')
            plt.legend()
            plt.savefig(f'd:/transformers-main/sample_{i+1}_{material_name}_result.png')
            plt.close()

def main():
    # 设置设备 - 添加更详细的GPU检查和诊断信息
    print("检查GPU可用性...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("GPU不可用，可能的原因:")
        print("1. 您的计算机没有NVIDIA GPU")
        print("2. CUDA驱动程序未正确安装")
        print("3. PyTorch没有安装CUDA版本")
        print("\n尝试安装支持CUDA的PyTorch版本:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n将使用CPU进行训练（速度较慢）")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        
        # 测试GPU是否正常工作
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
            test_result = test_tensor * 2
            print(f"GPU测试成功: {test_result}")
        except Exception as e:
            print(f"GPU测试失败: {e}")
            print("将回退到CPU训练")
            device = torch.device('cpu')
    
    # 修改数据文件路径为文件夹路径
    normal_folder_path = 'd:/transformers-main/NormalSpectra'
    distorted_folder_path = 'd:/transformers-main/DistortedSpectra'
    
    # 修复这里：正确接收所有返回值
    normal_spectra, distorted_spectra, currents, material_ids, spectra_scaler, current_scaler, intensity_ratios, material_map = load_and_preprocess_data(
        normal_folder_path, distorted_folder_path
    )
    
    # 保存物质映射和强度比例，用于推理时校正
    np.save('d:/transformers-main/intensity_ratios.npy', intensity_ratios)
    import json
    with open('d:/transformers-main/material_map.json', 'w') as f:
        json.dump(material_map, f)
    
    # 减少数据量，防止内存溢出
    # 如果数据量太大，随机抽样减少数据量
    if len(normal_spectra) > 1000:
        print(f"数据量较大 ({len(normal_spectra)}个样本)，按物质ID分层抽样...")
        
        # 获取唯一的物质ID
        unique_materials = np.unique(material_ids)
        num_materials = len(unique_materials)
        
        # 计算每种物质需要抽取的样本数
        samples_per_material = min(1000 // num_materials, 100)  # 每种物质最多100个样本
        
        # 分层抽样
        sampled_indices = []
        for material in unique_materials:
            material_indices = np.where(material_ids == material)[0]
            if len(material_indices) > samples_per_material:
                selected = np.random.choice(material_indices, size=samples_per_material, replace=False)
                sampled_indices.extend(selected)
            else:
                sampled_indices.extend(material_indices)
        
        # 转换为numpy数组并打乱顺序
        sampled_indices = np.array(sampled_indices)
        np.random.shuffle(sampled_indices)
        
        # 抽样
        normal_spectra = normal_spectra[sampled_indices]
        distorted_spectra = distorted_spectra[sampled_indices]
        currents = currents[sampled_indices]
        material_ids = material_ids[sampled_indices]
        
        print(f"抽样后数据量: {len(normal_spectra)}个样本")
    
    # 划分训练集、验证集和测试集 - 使用stratify参数确保每个集合中都有各种物质的样本
    X_train, X_test, y_train, y_test, c_train, c_test, m_train, m_test = train_test_split(
        distorted_spectra, normal_spectra, currents, material_ids, test_size=0.2, random_state=42, stratify=material_ids
    )
    
    X_train, X_val, y_train, y_val, c_train, c_val, m_train, m_val = train_test_split(
        X_train, y_train, c_train, m_train, test_size=0.25, random_state=42, stratify=m_train  # 0.25 x 0.8 = 0.2
    )
    
    # 创建数据集
    train_dataset = SpectralDataset(X_train, y_train, c_train, m_train)
    val_dataset = SpectralDataset(X_val, y_val, c_val, m_val)
    test_dataset = SpectralDataset(X_test, y_test, c_test, m_test)
    
    # 减小批量大小以减少内存使用
    batch_size = 16  # 从32减小到16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2 if device.type == 'cuda' else 0,
                             pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           num_workers=2 if device.type == 'cuda' else 0,
                           pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=1)  # 测试时一次处理一个样本
    
    # 获取物质数量
    num_materials = len(np.unique(material_ids))
    print(f"共有 {num_materials} 种不同物质")
    
    # 模型参数 - 减小模型大小以减少内存使用
    input_dim = 512  # 光谱通道数
    d_model = 128    # 从256减小到128
    nhead = 4        # 从8减小到4
    num_encoder_layers = 3  # 从4减小到3
    num_decoder_layers = 3  # 从4减小到3
    dim_feedforward = 512  # 从1024减小到512
    dropout = 0.2
    
    # 创建模型
    model = SpectralTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_materials=num_materials  # 添加物质数量参数
    ).to(device)  # 确保模型在正确的设备上
    
    # 定义损失函数和优化器
    criterion = SpectralLoss(intensity_weight=1.5, peak_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 添加训练函数
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None, accumulation_steps=1):
        model.to(device)
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # 添加训练进度跟踪
        from tqdm import tqdm
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            # 使用tqdm显示进度条
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            
            # 清空梯度
            optimizer.zero_grad()
            
            for i, batch in enumerate(progress_bar):
                distorted_spectrum = batch['distorted_spectrum'].to(device)
                current = batch['current'].to(device)
                normal_spectrum = batch['normal_spectrum'].to(device)
                material_id = batch['material_id'].to(device)
                
                # 前向传播
                output = model(distorted_spectrum, current, material_id)
                
                # 计算损失 - 传递电流值参数
                loss = criterion(output, normal_spectrum, current)
                
                # 缩放损失以适应梯度累积
                loss = loss / accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 添加进度条显示
                train_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.6f}'})
                
                # 每accumulation_steps步更新一次参数
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    optimizer.step()
                    
                    # 清空梯度
                    optimizer.zero_grad()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for batch in progress_bar:
                    distorted_spectrum = batch['distorted_spectrum'].to(device)
                    current = batch['current'].to(device)
                    normal_spectrum = batch['normal_spectrum'].to(device)
                    material_id = batch['material_id'].to(device)
                    
                    # 前向传播
                    output = model(distorted_spectrum, current, material_id)
                    
                    # 验证阶段也传递电流值参数
                    loss = criterion(output, normal_spectrum, current)
                    
                    val_loss += loss.item()
                    
                    # 更新进度条
                    progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 使用学习率调度器
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'd:/transformers-main/best_spectral_model.pth')
                print(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}')
            
            # 每个epoch后清理GPU缓存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return train_losses, val_losses
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=50,  # 减少训练轮数以加快训练
        device=device,
        scheduler=scheduler,
        accumulation_steps=2  # 使用梯度累积减少内存使用
    )
    
    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('d:/transformers-main/loss_curve.png')
    plt.close()
    
    # 修改visualize_results的调用
    visualize_results(
        model, 
        test_loader, 
        device, 
        spectra_scaler, 
        current_scaler,
        normal_spectra=y_test,
        currents=c_test,
        material_ids=m_test,
        intensity_ratios=intensity_ratios,
        material_map=material_map,
        num_samples=min(5, len(y_test))  # 可视化更多样本
    )
    
    # 保存归一化器和物质映射，用于后续推理
    import joblib
    joblib.dump(spectra_scaler, 'd:/transformers-main/spectra_scaler.pkl')
    joblib.dump(current_scaler, 'd:/transformers-main/current_scaler.pkl')
    
    print('训练完成，模型和归一化器已保存！')

if __name__ == '__main__':
    main()