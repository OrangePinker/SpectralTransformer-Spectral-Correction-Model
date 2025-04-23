import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        初始化光谱CNN模型
        
        参数:
            input_size: 输入光谱的长度
            num_classes: 类别数量
        """
        super(SpectralCNN, self).__init__()
        
        # 一维卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算全连接层的输入大小
        fc_input_size = 64 * (input_size // 4)
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 添加通道维度 [batch_size, input_size] -> [batch_size, 1, input_size]
        x = x.unsqueeze(1)
        
        # 卷积层
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class SpectralTransformer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SpectralTransformer, self).__init__()
        
        # 增加嵌入维度
        self.embedding_dim = 128
        
        # 线性投影层
        self.projection = nn.Linear(input_size, self.embedding_dim)
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_size, self.embedding_dim))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
        
        # 增加Transformer编码器层数和注意力头数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, 
            nhead=8,  # 增加注意力头数
            dim_feedforward=512,  # 增加前馈网络维度
            dropout=0.2  # 增加dropout防止过拟合
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)  # 增加层数
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x形状: [batch_size, input_size]
        
        # 添加序列维度
        x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        
        # 转置为Transformer输入形状
        x = x.transpose(0, 1)  # [1, batch_size, input_size]
        
        # 线性投影
        x = self.projection(x)  # [1, batch_size, embedding_dim]
        
        # 添加位置编码
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Transformer编码器
        x = self.transformer_encoder(x)  # [1, batch_size, embedding_dim]
        
        # 转置回来
        x = x.transpose(0, 1)  # [batch_size, 1, embedding_dim]
        
        # 全局平均池化
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, 1]
        x = self.global_avg_pool(x)  # [batch_size, embedding_dim, 1]
        x = x.squeeze(-1)  # [batch_size, embedding_dim]
        
        # 分类
        x = self.classifier(x)
        
        return x