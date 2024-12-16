import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def roll_n(X, axis, n):
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fft_conv2d(input, kernel):
    """
    Computes the convolution in the frequency domain given
    Expects input and kernel already in frequency domain!
    :param input: shape (B, Cin, H, W)
    :param kernel: shape (Cout, Cin, H, W)
    :param bias: shape of (B, Cout, H, W)
    :return:
    """
    input = torch.fft.fftn(input)
    kernel = torch.fft.fftn(kernel)

    # (a+bj)*(c+dj) = (ac-bd)+(ad+bc)j
    real = torch.real(input) * torch.real(kernel) - torch.imag(input) * torch.imag(kernel)
    im = torch.real(input) * torch.imag(kernel) + torch.imag(input) * torch.real(kernel)

    # Stack both channels and sum-reduce the input channels dimension
    out = torch.complex(real, im)

    out = torch.fft.ifftn(out).real
    return out


def get_wiener_matrix(psf, Gamma: float = 2e4, centre_roll: bool = False):
    """
    Get Weiner inverse of PSF
    :param psf: Point Spread Function (h,w)
    :param Gamma: Parameter for the Weiner filter
    :return: Weiner inverse
    """
    if centre_roll:
        for dim in range(2):
            psf = roll_n(psf, axis=dim, n=psf.size(dim) // 2)
    H = torch.fft.fftn(psf)
    H_conj = torch.conj(H)
    Habsq = torch.real(H * H_conj)
    W = torch.div(H_conj, (Habsq + Gamma))
    w = torch.fft.ifftn(W).real  #dp不支持复数
    return w


class FFTLayer(nn.Module):
    def __init__(self, initial_mode='calibration', fft_gamma=2e4, image_size=(256, 256), is_require_grad=True):
        super().__init__()
        self.fft_gamma = fft_gamma
        self.image_size = image_size
        self.initial_mode = initial_mode
        self.is_require_grad = is_require_grad
        ###
        # psf = torch.rand(1052, 1400).to(self.device)
        # 加载图像
        psf = Image.open(r"D:\cqy\phlat_data\psf\psf_14cm_15um_100exposure.png").convert('L')
        # 将图像转换为张量
        transform = transforms.ToTensor()  # 转换为Tensor并归一化到[0, 1]
        psf = transform(psf)
        psf = psf.squeeze(0)

        wiener_matrix = get_wiener_matrix(
            psf, Gamma=self.fft_gamma, centre_roll=False
        )
        # wiener_matrix = torch.rand(1280,1408).to(self.device)

        self.wiener_matrix = nn.Parameter(wiener_matrix, requires_grad=self.is_require_grad)

        self.normalizer = nn.Parameter(
            torch.tensor([1 / 0.0008]).reshape(1, 1, 1, 1), requires_grad=self.is_require_grad)


    def forward(self, img):
        fft_layer = 1 * self.wiener_matrix

        # Centre roll
        for dim in range(2):
            fft_layer = roll_n(
                fft_layer, axis=dim, n=fft_layer.size(dim) // 2
            )

        # Make 1 x 1 x H x W
        fft_layer = fft_layer.unsqueeze(0).unsqueeze(0)

        # FFT Layer dims
        _, _, fft_h, fft_w = fft_layer.shape

        # Target image dims
        img_h = self.image_size[0]
        img_w = self.image_size[1]

        # Do FFT convolve
        img = fft_conv2d(img, fft_layer) * self.normalizer

        # Centre Crop
        img = img[
            :,
            :,
            fft_h // 2 - img_h // 2 : fft_h // 2 + img_h // 2,
            fft_w // 2 - img_w // 2 : fft_w // 2 + img_w // 2,
        ]

        img =  F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)

        return img



if __name__ == '__main__':
    import os
    # 设置使用的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU 6 和 GPU 7
    device = torch.device('cuda')

    # 初始化模型并使用 DataParallel 包装
    model = FFTLayer().to(device)
    model = torch.nn.DataParallel(model)  # 包装以支持多卡

    # 创建输入数据
    img = torch.rand(8, 3, 1052, 1400).to(device)

    # 前向传播
    y = model(img)
    print(y.shape)
