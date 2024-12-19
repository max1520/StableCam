from ldm.modules.conv_modules.modules import LayerNorm, Block
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

class SR_Net(nn.Module):
    def __init__(self, in_chans: int = 3, loss_sr_weight=1.0,
                 channels=64, layer_scale_init_value: float = 1e-6, dropout=0
                 ):
        super().__init__()
        self.loss_sr_weight = loss_sr_weight
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_chans, channels, kernel_size=3, stride=1, padding=1),
            normalization(channels)
        )
        # 添加两个 Block
        self.blocks = nn.Sequential(
            Block(channels, layer_scale_init_value=layer_scale_init_value, dropout=dropout),
            Block(channels, layer_scale_init_value=layer_scale_init_value, dropout=dropout)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda')
    sr_block = SR_Net().to(device)
    x = torch.rand(1, 3, 512, 512).to(device)
    y = sr_block(x)
    print(y.shape)  #(1,3,512,512)