import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class My_CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(My_CNN, self).__init__()

        # ---- 卷积层 + 批归一化：逐层提取特征，通道数 3→32→64→128→256 ----
        # Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # padding=1 保持特征图空间尺寸不变（same padding），BN 加速收敛、稳定梯度
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 2×2 最大池化：每次将特征图尺寸减半（224→112→56→28→14）
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout 随机失活 30% 神经元，防止过拟合
        self.dropout = nn.Dropout(0.3)

        # ---- 全连接层：将卷积特征映射到分类结果 ----
        # 经过4次池化后特征图尺寸为 14×14，通道数 256，展平后 256×14×14 = 50176
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        # 输出层：128 维 → num_classes 维（WBC/RBC/Platelets 三类）
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=3, pretrained=True):
        super(ResNet, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        else:
            self.model = models.resnet50(pretrained=pretrained)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = ResNet(num_classes=3, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)

    model2 = My_CNN(num_classes=3)
    out2 = model2(x)
    print(out2.shape)