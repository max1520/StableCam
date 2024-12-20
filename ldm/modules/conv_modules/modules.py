import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class Block(nn.Module):
    def __init__(self, channels, layer_scale_init_value=1e-6, dropout=0):
        super().__init__()

        # 构造输入层 in_layers
        self.in_layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            normalization(channels),
        )

        # 构造输出层 out_layers
        self.out_layers = nn.Sequential(
            nn.Conv2d(channels, 4 * channels, kernel_size=1, stride=1),
            nn.SiLU(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, stride=1),
            normalization(channels),
        )

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, channels, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop = nn.Dropout(p=dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.in_layers(x)
        x = self.out_layers(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = shortcut + self.drop(x)
        return x

class Feature_Block(TimestepBlock):
    def __init__(self,
                 channels=256,
                 emb_channels=1024,
                 layer_scale_init_value=1e-6,
                 dropout=0,
                 out_channels=None,
                 ):
        super().__init__()

        self.out_channels = out_channels or channels

        # 构造输入层 in_layers
        self.in_layers = nn.Sequential(
            # 判断是否使用深度可分离卷积
            nn.Conv2d(channels, self.out_channels, kernel_size=7, padding=3, groups=channels)
            if self.out_channels % channels == 0 else nn.Conv2d(channels, self.out_channels, kernel_size=3, padding=1),
            normalization(self.out_channels),  # 假设 normalization 是你定义的正则化层
        )

        # 构造嵌入层 emb_layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, self.out_channels),
        )

        # 构造输出层 out_layers
        self.out_layers = nn.Sequential(
            nn.Conv2d(self.out_channels, 4 * self.out_channels, kernel_size=1, stride=1),
            nn.SiLU(),
            nn.Conv2d(4 * self.out_channels, self.out_channels, kernel_size=1, stride=1),
            normalization(self.out_channels),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, self.out_channels, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # 通过 emb_layers 处理嵌入
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]

        # 输入处理
        shortcut = x
        x = self.in_layers(x)

        # 添加嵌入输出
        x = x + emb_out

        # 通过输出层处理
        x = self.out_layers(x)

        # gamma 调整
        if self.gamma is not None:
            x = self.gamma * x

        # 最终处理
        shortcut = self.skip_connection(shortcut)
        x = shortcut + self.drop(x)

        return x

if __name__ == '__main__':
    device = torch.device('cuda')
    # model = Feature_Block(
    #     channels=256,
    #     emb_channels=1024,
    #     layer_scale_init_value=1e-6,
    #     dropout=0,
    #     out_channels=None,
    # ).to(device)
    model2 = Block(channels=256).to(device)
    x = torch.rand(1,256,64,64).to(device)
    emb = torch.rand(1,1024).to(device)
    # y = model(x, emb)
    y= model2(x)
    print(y.shape)  #(1,256,64,64)