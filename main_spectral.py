import os
import argparse
import torch
from spectral_dataset import load_spectral_data, plot_first_column_data
from spectral_model import SpectralCNN, SpectralTransformer
from train_eval import train_model, evaluate_model, plot_confusion_matrix, plot_training_history, plot_class_accuracies, save_results_to_csv

def main(args):
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘制原始数据和预处理后的数据对比图
    print("绘制光谱数据对比图...")
    plot_first_column_data(args.data_folder, args.background_file, args.output_dir)
    
    # 加载数据
    print(f"从 {args.data_folder} 加载光谱数据...")
    train_loader, val_loader, test_loader, class_names = load_spectral_data(
        args.data_folder,
        args.background_file,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        random_state=args.seed
    )
    
    # 获取输入大小（光谱长度）
    sample_data, _ = next(iter(train_loader))
    input_size = sample_data.shape[1]
    num_classes = len(class_names)
    
    print(f"光谱长度: {input_size}")
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {class_names}")
    
    # 创建模型
    if args.model_type == 'cnn':
        model = SpectralCNN(input_size, num_classes)
        print("使用CNN模型")
    else:
        model = SpectralTransformer(input_size, num_classes)
        print("使用Transformer模型")
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置模型保存路径
    model_save_path = None
    if args.save_model:
        if args.model_path:
            model_save_path = args.model_path
        elif args.output_dir:
            model_save_path = os.path.join(args.output_dir, f'{args.model_type}_model.pth')
        else:
            model_save_path = f'{args.model_type}_model.pth'
    
    # 绘制原始数据和预处理后的数据对比图
    print("绘制光谱数据对比图...")
    plot_first_column_data(args.data_folder, args.background_file, args.output_dir)
    
    # 加载数据
    print(f"从 {args.data_folder} 加载光谱数据...")
    train_loader, val_loader, test_loader, class_names = load_spectral_data(
        args.data_folder,
        args.background_file,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        random_state=args.seed
    )
    
    # 获取输入大小（光谱长度）
    sample_data, _ = next(iter(train_loader))
    input_size = sample_data.shape[1]
    num_classes = len(class_names)
    
    print(f"光谱长度: {input_size}")
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {class_names}")
    
    # 创建模型
    if args.model_type == 'cnn':
        model = SpectralCNN(input_size, num_classes)
        print("使用CNN模型")
    else:
        model = SpectralTransformer(input_size, num_classes)
        print("使用Transformer模型")
    
    # 训练模型
    print(f"\n开始训练 {args.model_type} 模型...")
    model, history = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, 
        learning_rate=args.lr,
        device=device,
        save_path=model_save_path
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存训练历史曲线
    history_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=history_path)
    
    # 评估模型
    print("评估模型...")
    accuracy, conf_matrix, class_accuracies, all_preds, all_labels = evaluate_model(
        model, test_loader, class_names, device
    )
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(conf_matrix, class_names, save_path=cm_path)
    
    # 绘制类别准确率
    acc_path = os.path.join(args.output_dir, 'class_accuracy.png')
    plot_class_accuracies(class_accuracies, class_names, save_path=acc_path)
    
    # 保存结果到CSV
    csv_path = os.path.join(args.output_dir, 'classification_results.csv')
    save_results_to_csv(class_names, class_accuracies, accuracy, csv_path)
    
    # 保存模型
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'spectral_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到 {model_path}")
    
    print(f"所有结果已保存到 {args.output_dir} 目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='光谱数据分类')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='包含Excel文件的文件夹路径')
    parser.add_argument('--background_file', type=str, required=True,
                        help='背景数据Excel文件路径')
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['cnn', 'transformer'],
                        help='模型类型: cnn 或 transformer (默认: transformer)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数 (默认: 30)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='学习率 (默认: 0.0005)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='禁用CUDA')
    parser.add_argument('--save_model', action='store_true',
                        help='是否保存训练好的模型')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存路径，默认为output_dir/model.pth')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (默认: ./output)')
    
    args = parser.parse_args()
    main(args)