# LightTrack
基于STARK-lightning轻量模型进行改进，现共提出三种改进方案并进行实验
1.深度可分离卷积的替换尝试
参考LightTrack方法在模型头部中普遍使用了轻量常用的DSConv，相比之下，STARK-lightning模型所用头部由若干RepVGG块（参数量较多）与一层3×3卷积构成。
本方案将RepVGG用等量（2）或更多数量（3-6）的深度可分离卷积块做替换，主要改进点如下：
1.使用深度可分离卷积（DepthwiseSeparableConv）替换标准卷积层，以减少参数量和计算量。
2.减少卷积层的通道数（从 128 减少到 64）。
3.使用 ReLU6 代替 ReLU 作为激活函数，进一步优化计算效率。又因为它的输出范围有限，更容易进行定点数表示，所以在移动设备或嵌入式设备等量化模型中会有更佳的表现。
代码如下：
'''python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.relu(x)
        return x

class Corner_Predictor_Lite_Rep_v2(nn.Module):
    """ Corner Predictor module (Lite version with repvgg style)"""

    def __init__(self, inplanes=128, channel=64, feat_sz=20, stride=16):
        super(Corner_Predictor_Lite_Rep_v2, self).__init__()
        self.feat_sz = feat_sz
        self.feat_len = feat_sz ** 2
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''convolution tower for two corners'''
        self.conv_tower = nn.Sequential(
            DepthwiseSeparableConv(inplanes, channel, kernel_size=3, padding=1),
            DepthwiseSeparableConv(channel, channel, kernel_size=3, padding=1),
            nn.Conv2d(channel, 2, kernel_size=1)
        )

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = (torch.arange(0, self.feat_sz).view(-1, 1) + 0.5) * self.stride  # here we can add a 0.5
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        score_map = self.conv_tower(x)  # (B,2,H,W)
        return score_map[:, 0, :, :], score_map[:, 1, :, :]

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_len))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y
'''
