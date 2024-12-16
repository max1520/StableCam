import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = 0.1 * shortcut + x
        return x

class SR_Net(nn.Module):
    def __init__(self, in_chans: int = 3, loss_sr_weight=1.0,
                 dims=64,  layer_scale_init_value: float = 1e-6,
                 ):
        super().__init__()
        self.loss_sr_weight = loss_sr_weight
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_chans, dims, kernel_size=3, stride=1, padding=1),
            LayerNorm(dims, eps=1e-6, data_format="channels_first")
        )
        # 添加两个 Block
        self.blocks = nn.Sequential(
            Block(dim=dims, layer_scale_init_value=layer_scale_init_value),
            Block(dim=dims, layer_scale_init_value=layer_scale_init_value)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda')
    sr_block = SR_Net().to(device)
    x = torch.rand(1,3,512,512).to(device)
    y = sr_block(x)
    print(y.shape)  #(1,3,512,512)