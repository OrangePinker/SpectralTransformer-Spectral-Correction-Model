import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os

def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.0005, device='cuda', save_path=None):
    """
    训练模型
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
        save_path: 模型保存路径，如果提供则保存模型
        
    返回:
        model: 训练好的模型
        history: 包含训练和验证损失的字典
    """
    model = model.to(device)
    # 使用标签平滑的交叉熵损失函数，减少过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 使用AdamW优化器，更好的权重衰减实现
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 使用余弦退火学习率调度器，更平滑的学习率变化
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 用于存储训练历史
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # 早停机制
    best_val_acc = 0.0  # 改为监控验证准确率而不是损失
    patience = 15
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算平均训练损失和准确率
        epoch_train_loss = train_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)
        
        train_accuracy = correct / total
        train_acc_history.append(train_accuracy)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算平均验证损失和准确率
        epoch_val_loss = val_loss / len(val_loader)
        val_loss_history.append(epoch_val_loss)
        
        val_accuracy = correct / total
        val_acc_history.append(val_accuracy)
        
        # 学习率调度器步进
        scheduler.step()
        
        # 早停检查 - 基于验证准确率
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f'Epoch {epoch+1}: 新的最佳验证准确率: {val_accuracy:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停! 验证准确率在{patience}个epoch内未改善。最佳准确率: {best_val_acc:.4f}')
                model.load_state_dict(best_model_state)
                break
        
        # 打印每个epoch的损失和验证准确率
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.6f}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
    
    # 如果早停未触发，加载最佳模型
    if epoch == num_epochs - 1 and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'训练完成! 加载最佳模型，验证准确率: {best_val_acc:.4f}')
    
    # 保存模型
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 保存模型
        torch.save(model.state_dict(), save_path)
        print(f'模型已保存到: {save_path}')
    
    return model, {'train_loss': train_loss_history, 'val_loss': val_loss_history, 
                  'train_acc': train_acc_history, 'val_acc': val_acc_history}

def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    参数:
        history: 包含训练和验证损失的字典
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 检查键是否存在，如果不存在则跳过绘制
    if 'train_acc' in history and 'val_acc' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    else:
        print("警告: 训练历史中缺少准确率数据，只绘制损失曲线")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    plt.close()

def evaluate_model(model, test_loader, class_names, device='cuda'):
    """
    评估模型并生成混淆矩阵
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        class_names: 类别名称列表
        device: 评估设备
        
    返回:
        accuracy: 准确率
        conf_matrix: 混淆矩阵
        class_accuracies: 每个类别的准确率
        all_preds: 所有预测结果
        all_labels: 所有真实标签
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # 计算每个类别的准确率
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    # 打印分类报告
    print(f"模型准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 打印每个类别的准确率
    for i, acc in enumerate(class_accuracies):
        print(f"{class_names[i]} 类别准确率: {acc:.4f}")
    
    return accuracy, conf_matrix, class_accuracies, all_preds, all_labels

def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
        conf_matrix: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在每个单元格中显示数字
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")
    plt.close()

def plot_class_accuracies(class_accuracies, class_names, save_path=None):
    """
    绘制各类别准确率
    
    参数:
        class_accuracies: 各类别准确率列表
        class_names: 类别名称列表
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracies * 100)
    
    # 在柱状图上方显示准确率值
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc*100:.1f}%', ha='center', va='bottom')
    
    plt.title('Accuracy by Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)  # 设置y轴范围，留出空间显示文本
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class accuracy plot saved to: {save_path}")
    plt.close()

def save_results_to_csv(class_names, class_accuracies, accuracy, save_path):
    """
    将结果保存到CSV文件
    
    参数:
        class_names: 类别名称列表
        class_accuracies: 每个类别的准确率
        accuracy: 总体准确率
        save_path: 保存路径
    """
    import pandas as pd
    
    # 创建结果数据框
    results = {'Class': class_names, 'Accuracy': class_accuracies}
    df = pd.DataFrame(results)
    
    # 添加总体准确率
    df = df.append({'Class': 'Overall', 'Accuracy': accuracy}, ignore_index=True)
    
    # 保存到CSV
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"分类结果已保存到 {save_path}")