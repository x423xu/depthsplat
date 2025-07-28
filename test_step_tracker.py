import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 定义网络结构（基于您提供的ExampleNetwork）
class ExampleNetwork(ME.MinkowskiNetwork):
    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)

# 创建模拟数据集类
class PointCloudDataset(Dataset):
    def __init__(self, num_samples=100, num_points=200, num_classes=5):
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机坐标 (batch_size, num_points, 3)
        coords = torch.randint(0, 100, (2*self.num_points, 3))
        coords = coords[:self.num_points]  # 只取前num_points个点
        
        # 添加batch索引 (batch_idx, x, y)
        coords = torch.cat([torch.full((self.num_points, 1), idx), coords], dim=1)
        
        # 生成随机特征 (batch_size, num_points, 3)
        features = torch.rand((self.num_points, 3))
        
        # 生成随机标签 (batch_size)
        label = torch.randint(0, self.num_classes, (1,))
        
        return coords, features, label

# 创建数据加载器
def create_data_loader(batch_size=4):
    dataset = PointCloudDataset(num_samples=100, num_points=200)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ME.utils.batch_sparse_collate)

# 训练函数
def train_model():
    # 初始化设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ExampleNetwork(in_feat=3, out_feat=5, D=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    data_loader = create_data_loader()
    
    print(f"Network architecture:\n{net}")
    print(f"Using device: {device}")
    
    # 训练循环
    for epoch in range(5):  # 训练5个epoch
        net.train()
        running_loss = 0.0
        
        for batch_idx, (coords, feat, label) in enumerate(data_loader):
            # 将数据移至设备
            coords, feat, label = coords.to(device), feat.to(device), label.to(device)
            
            # 创建稀疏张量 [3](@ref)
            input = ME.SparseTensor(
                features=feat,
                coordinates=coords,
                device=device
            )
            
            # 前向传播
            optimizer.zero_grad()
            output = net(input)
            
            # 计算损失
            loss = criterion(output.F, label)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/5], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
        
        # 验证精度
        accuracy = validate(net, device)
        print(f'Epoch [{epoch+1}/5] completed. Avg Loss: {running_loss/len(data_loader):.4f}, Val Accuracy: {accuracy:.2%}\n')
    
    print("Training completed!")

# 验证函数
def validate(model, device):
    model.eval()
    data_loader = create_data_loader(batch_size=4)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for coords, feat, label in data_loader:
            coords, feat, label = coords.to(device), feat.to(device), label.to(device)
            
            # 创建稀疏张量 [6](@ref)
            input = ME.SparseTensor(
                features=feat,
                coordinates=coords,
                device=device
            )
            
            output = model(input)
            _, predicted = torch.max(output.F, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    return correct / total

# 执行训练
if __name__ == "__main__":
    # 设置随机种子确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查MinkowskiEngine是否可用 [1](@ref)
    print("MinkowskiEngine version:", ME.__version__)
    
    # 开始训练
    train_model()