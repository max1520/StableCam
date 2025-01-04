from ldm.modules.conv_modules.modules import Block
import torch.nn as nn
import torch
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class SR_Block(nn.Module):
    def __init__(self,
                 in_chans: int = 3,
                 out_chans: int = 3,
                 channels=64,
                 num_blocks: int = 2,
                 loss_sr_weight=1.0,
                 layer_scale_init_value: float = 1e-6,
                 dropout=0,
                 ):
        super().__init__()
        self.loss_sr_weight = loss_sr_weight
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_chans, channels, kernel_size=3, stride=1, padding=1),
            normalization(channels)
        )
        # 根据 num_blocks 动态堆叠 Block
        self.blocks = nn.Sequential(
            *[Block(channels, layer_scale_init_value=layer_scale_init_value, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_conv = nn.Conv2d(channels, out_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda')
    sr_block = SR_Block().to(device)
    x = torch.rand(1, 3, 512, 512).to(device)
    y = sr_block(x)
    print(y.shape)  #(1,3,512,512)