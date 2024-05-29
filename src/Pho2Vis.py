import torch
import torch.nn as nn
from torchvision import models

class Pho2Vis(nn.Module):
    def __init__(self,numerical_features=3):
        super(Pho2Vis, self).__init__()
        # 数值特征的维度
        self.numerical_features = numerical_features
        # 加载预训练的InceptionV3模型,冻结前面的层
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        for param in self.inception.parameters():
            param.requires_grad = False
        # 移除辅助输出和全连接层
        self.inception.fc = nn.Identity()
        self.inception.aux_logits = False

        # 1x1卷积层，用于将两个特征图在通道维度上堆叠
        self.conv1x1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(1024 + self.numerical_features, 512)  # 假设数值特征有3个维度
        self.fc2 = nn.Linear(512 + self.numerical_features, 512)
        self.fc3 = nn.Linear(512, 1)

        # 激活函数和最终的缩放倍数
        self.sigmoid = nn.Sigmoid()
        self.scale = 15000

    def forward(self, image0, image1, numerical_features):
        # 对两个图像使用相同的InceptionV3模型提取特征
        features0 = self.inception(image0)
        features1 = self.inception(image1)

        # 增加一个维度，以便在通道维度上堆叠
        features0 = features0.unsqueeze(1)
        features1 = features1.unsqueeze(1)

        # 将特征图在通道维度上堆叠
        # print(features0.shape, features1.shape)
        combined_features = torch.cat((features0, features1), dim=1)
        # combined_features = combined_features.permute(0, 2, 1)
        # print(combined_features.shape)
        combined_features = self.conv1x1(combined_features)
        combined_features = combined_features.squeeze(1)
        
        # 平坦化特征图
        combined_features = torch.flatten(combined_features, start_dim=1)

        # 拼接数值特征
        full_features = torch.cat((combined_features, numerical_features), dim=1)

        # 通过全连接层
        x = self.fc1(full_features)
        x = torch.relu(x)
        x = torch.cat((x, numerical_features), dim=1)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        # 应用Sigmoid激活函数并缩放结果
        x = self.sigmoid(x) * self.scale

        return x

# 示例初始化和前向传递
if __name__ == "__main__":
    # 假设有用的示例数据
    batch_size = 10
    image = torch.rand(batch_size, 3, 299, 299)  # InceptionV3需要299x299的输入
    numerical_features = torch.rand(batch_size, 3)  # 假设有3个数值特征
    print(image.shape, numerical_features.shape)
    model = Pho2Vis()
    output = model(image, image, numerical_features)
    print(output.shape)  # 应该输出 (batch_size, 1)
    # print(model)  # 打印模型结构