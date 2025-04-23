import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
from spectrum_correction import SpectralTransformer

def load_model(model_path, model_params):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型权重文件路径
        model_params: 模型参数字典
    
    返回:
        加载好权重的模型
    """
    model = SpectralTransformer(
        input_dim=model_params['input_dim'],
        d_model=model_params['d_model'],
        nhead=model_params['nhead'],
        num_encoder_layers=model_params['num_encoder_layers'],
        num_decoder_layers=model_params['num_decoder_layers'],
        dim_feedforward=model_params['dim_feedforward'],
        dropout=model_params['dropout'],
        num_materials=model_params.get('num_materials', 1)  # 添加物质数量参数，默认为1
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

def correct_spectrum(model, distorted_spectrum, current, spectra_scaler, current_scaler, 
                    intensity_ratio=None, normal_spectra=None, currents=None, material_ids=None, material_id=0):
    """
    对畸变光谱进行矫正
    
    参数:
        model: 训练好的模型
        distorted_spectrum: 畸变光谱数据 [通道数]
        current: 管电流值
        spectra_scaler: 光谱归一化器
        current_scaler: 电流归一化器
        intensity_ratio: 强度校正比例（可选）
        normal_spectra: 正常光谱数据，用于参考总光子数 [样本数, 通道数]
        currents: 所有样本的电流值，用于建立电流-光子数关系 [样本数]
        material_ids: 所有样本的物质ID [样本数]
        material_id: 当前样本的物质ID，默认为0
    
    返回:
        校正后的光谱数据
    """
    # 归一化输入数据
    distorted_spectrum_scaled = spectra_scaler.transform(distorted_spectrum.reshape(1, -1))
    current_scaled = current_scaler.transform(np.array([[current]]))
    
    # 转换为张量
    distorted_tensor = torch.tensor(distorted_spectrum_scaled, dtype=torch.float32)
    current_tensor = torch.tensor(current_scaled, dtype=torch.float32)
    material_id_tensor = torch.tensor([material_id], dtype=torch.long)
    
    # 使用模型进行推理 - 添加物质ID参数
    with torch.no_grad():
        corrected_spectrum_scaled = model(distorted_tensor, current_tensor, material_id_tensor)
    
    # 反归一化输出数据
    corrected_spectrum = spectra_scaler.inverse_transform(corrected_spectrum_scaled.numpy())
    
    # 确保输出非负
    corrected_spectrum = np.clip(corrected_spectrum, 0, None)
    
    # 应用强度校正（如果提供）
    if intensity_ratio is not None:
        # 计算当前总强度
        distorted_sum = np.sum(distorted_spectrum)
        corrected_sum = np.sum(corrected_spectrum[0])
        
        # 应用校正比例
        target_sum = distorted_sum * intensity_ratio
        if corrected_sum > 0:  # 避免除零错误
            scale_factor = target_sum / corrected_sum
            # 限制最大缩放比例，避免过度放大
            scale_factor = min(scale_factor, 1.2)
            corrected_spectrum[0] = corrected_spectrum[0] * scale_factor
    
    # 将结果转换为整数
    corrected_spectrum = np.round(corrected_spectrum).astype(int)
    
    return corrected_spectrum[0]

def load_material_map(map_path):
    """
    加载物质映射信息
    
    参数:
        map_path: 物质映射文件路径
    
    返回:
        物质映射字典
    """
    try:
        with open(map_path, 'r') as f:
            material_map = json.load(f)
        return material_map
    except Exception as e:
        print(f"加载物质映射时出错: {e}")
        return {}

def main():
    # 模型和数据路径
    model_path = 'd:\\transformers-main\\best_spectral_model.pth'
    spectra_scaler_path = 'd:\\transformers-main\\spectra_scaler.pkl'
    current_scaler_path = 'd:\\transformers-main\\current_scaler.pkl'
    distorted_folder_path = 'd:\\transformers-main\\DistortedSpectra'  # 修改为文件夹路径
    material_map_path = 'd:\\transformers-main\\material_map.json'
    output_folder = 'd:\\transformers-main\\CorrectedSpectra'
    
    # 尝试加载强度比例数据
    try:
        intensity_ratios = np.load('d:\\transformers-main\\intensity_ratios.npy')
        print("已加载强度比例数据")
        # 计算平均强度比例
        mean_intensity_ratio = np.mean(intensity_ratios)
        print(f"平均强度比例: {mean_intensity_ratio:.4f}")
    except Exception as e:
        print(f"加载强度比例数据时出错: {e}")
        print("将不进行强度校正")
        mean_intensity_ratio = None
    
    # 加载物质映射
    material_map = load_material_map(material_map_path)
    if not material_map:
        print("警告: 未找到物质映射或加载失败，将使用数字ID作为物质标识")
        # 创建一个默认的映射，确保有9种物质
        material_map = {str(i): f"物质_{i}" for i in range(9)}
    
    print(f"物质映射包含 {len(material_map)} 种物质:")
    for material_id, material_name in material_map.items():
        print(f"  ID {material_id}: {material_name}")
    
    # 修改模型参数 - 使其与训练时一致
    model_params = {
        'input_dim': 512,      # 光谱通道数
        'd_model': 128,        # 从256改为128
        'nhead': 4,            # 从8改为4
        'num_encoder_layers': 3, # 从4改为3
        'num_decoder_layers': 3, # 从4改为3
        'dim_feedforward': 512, # 从1024改为512
        'dropout': 0.2,        # 保持不变
        'num_materials': len(material_map)  # 物质数量
    }
    
    try:
        # 加载模型和归一化器
        print("正在加载模型和归一化器...")
        model = load_model(model_path, model_params)
        spectra_scaler = joblib.load(spectra_scaler_path)
        current_scaler = joblib.load(current_scaler_path)
        print("模型和归一化器加载成功")
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取DistortedSpectra文件夹中的所有Excel文件
        excel_files = [f for f in os.listdir(distorted_folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]
        
        if not excel_files:
            print(f"错误: 在 {distorted_folder_path} 中未找到Excel文件")
            return
        
        print(f"找到 {len(excel_files)} 个Excel文件需要处理")
        
        # 处理每个Excel文件 - 每个文件代表一种物质
        for excel_file in excel_files:
            file_path = os.path.join(distorted_folder_path, excel_file)
            print(f"\n正在处理文件: {excel_file}")
            
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path, header=None)
                
                # 提取管电流值（第一行）和光谱数据（第2行到第513行）
                currents = df.iloc[0, :].values
                spectra = df.iloc[1:513, :].values
                
                # 创建结果DataFrame，使用与输入相同的形状
                result_df = pd.DataFrame(np.zeros((513, spectra.shape[1])))
                
                # 填充第一行（管电流值）
                result_df.iloc[0, :] = currents
                
                # 从文件名中提取物质ID（假设文件名格式为"物质_X.xlsx"或类似格式）
                material_id = 0  # 默认物质ID
                
                # 尝试从文件名中提取物质ID
                for id_str, name in material_map.items():
                    if name.lower() in excel_file.lower():
                        material_id = int(id_str)
                        print(f"  根据文件名识别为物质ID: {material_id} ({name})")
                        break
                
                # 对每个样本进行处理
                for col in range(spectra.shape[1]):
                    distorted_spectrum = spectra[:, col]
                    current = currents[col]
                    
                    # 直接使用识别的物质ID进行校正
                    corrected_spectrum = correct_spectrum(
                        model, 
                        distorted_spectrum, 
                        current, 
                        spectra_scaler, 
                        current_scaler, 
                        mean_intensity_ratio,
                        material_id=material_id
                    )
                    
                    # 将结果保存到DataFrame（从第2行开始）
                    result_df.iloc[1:513, col] = corrected_spectrum
                    
                    if col % 10 == 0 or col == spectra.shape[1] - 1:
                        print(f"  处理进度: {col+1}/{spectra.shape[1]}")
                
                # 保存校正后的结果到输出文件夹，保持相同的文件名
                output_path = os.path.join(output_folder, excel_file)
                result_df.to_excel(output_path, header=False, index=False)
                print(f"已保存校正结果到: {output_path}")
                
                # 生成可视化结果
                # 随机选择最多5个样本进行可视化
                num_samples = min(5, spectra.shape[1])
                sample_indices = np.random.choice(spectra.shape[1], num_samples, replace=False)
                
                for i, idx in enumerate(sample_indices):
                    current = currents[idx]
                    distorted_spectrum = spectra[:, idx]
                    corrected_spectrum = result_df.iloc[1:513, idx].values
                    
                    # 可视化
                    plt.figure(figsize=(12, 6))
                    plt.plot(distorted_spectrum, label='畸变光谱', alpha=0.7)
                    plt.plot(corrected_spectrum, label='校正光谱', alpha=0.7)
                    plt.title(f'{os.path.splitext(excel_file)[0]} - 电流值: {current:.4f}')
                    plt.xlabel('通道[1-512]')
                    plt.ylabel('光子计数')
                    plt.legend()
                    plt.savefig(f'd:\\transformers-main\\CorrectedSpectra\\{os.path.splitext(excel_file)[0]}_sample_{i+1}.png')
                    plt.close()
                
                print(f"已为 {excel_file} 生成可视化结果")
                
            except Exception as e:
                print(f"处理文件 {excel_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n所有处理完成！")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()