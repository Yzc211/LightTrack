import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的卷积层封装
def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class CornerDetector(nn.Module):
    def __init__(self, inplanes, channel):
        super().__init__()
        self.conv_tower = nn.Sequential(
            conv(inplanes, channel),
            conv(channel, channel // 2),
            conv(channel // 2, channel // 4),
            conv(channel // 4, channel // 8),
            nn.Conv2d(channel // 8, 2, kernel_size=3, padding=1)  # 输出两个通道的得分图
        )

    def get_score_map(self, x):
        score_map = self.conv_tower(x)  # (B, 2, H, W)
        return score_map[:, 0, :, :], score_map[:, 1, :, :]

    def forward(self, x):
        score_map_tl, score_map_br = self.get_score_map(x)
        # 这里可以添加后续处理逻辑
        return score_map_tl, score_map_br

# 示例
model = CornerDetector(inplanes=3, channel=64)
x = torch.randn(2, 3, 32, 32)  # 输入特征 (B=2, C=3, H=32, W=32)
score_map_tl, score_map_br = model(x)

print("Score Map TL shape:", score_map_tl.shape)  # (2, 32, 32)
print("Score Map BR shape:", score_map_br.shape)  # (2, 32, 32)
print("Score Map TL:", score_map_tl)  # (2, 32, 32)
print("Score Map BR:", score_map_br)  # (2, 32, 32)
