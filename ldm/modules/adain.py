import torch
import torch.nn as nn


class AdaIn(nn.Module):
    '''
    Adaptive Instance Normalization
    '''

    def __init__(self, nchannel, dim):
        """
        :param nchannel: content's channels
        :param dim: style's dimension
        """
        super().__init__()
        # content (channels)
        self.norm = nn.InstanceNorm2d(nchannel)
        # style   (dim, channels*2)
        self.style_transform = nn.Linear(dim, nchannel * 2)

    def forward(self, content, style):
        # 对style通过一个FC得到缩放系数和偏置系数
        # style 输入维度是 (batch_size, nchannel)
        factor, bias = self.style_transform(style).chunk(2, 1)  # (batch_size, nchannel * 2)

        # 将 factor 和 bias 调整为适配 img_feat 的维度
        factor = factor.unsqueeze(2).unsqueeze(3)  # (batch_size, nchannel, 1, 1)
        bias = bias.unsqueeze(2).unsqueeze(3)  # (batch_size, nchannel, 1, 1)

        # 对内容图像特征进行 InstanceNorm
        content = self.norm(content)

        # 应用缩放系数和偏置系数
        result = content * factor + bias
        return result

if __name__ == '__main__':
    # Example usage:
    nchannel = 512
    dim = 768
    adain = AdaIn(nchannel, dim)
    img_feat = torch.randn(16, nchannel, 64, 64)  # Example feature map
    style = torch.randn(16, dim)  # Example style vector

    output = adain(img_feat, style)