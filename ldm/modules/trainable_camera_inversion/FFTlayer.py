import torch
import torch.nn as nn
import torch.fft
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image


# 定义 FFT Layer 作为 nn.Module 类
class FFTlayer(nn.Module):
    def __init__(self, psf_image_path, gamm_l2=1e4, crop_area=None):
        if crop_area is None:
            crop_area = [618, 437, 200, 200]
        super().__init__()
        # 加载 PSF 图像
        psf_image = Image.open(psf_image_path).convert('L')
        psf_tensor = transforms.ToTensor()(psf_image).unsqueeze(0)  # 转为张量并增加 batch 维度
        self.psf = nn.Parameter(psf_tensor, requires_grad=True)  # 将 PSF 定义为不可训练的参数
        self.normalizer = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)

        # 正则化常数
        self.gamm_l2 = gamm_l2

        # 裁剪区域
        self.crop_area = crop_area

    def forward(self, x):
        """
        输入 x 维度为 (b, 3, H, W)，PSF 维度为 (1, 1, H, W)
        """
        # 将三通道输入转换为单通道（取均值）
        x = torch.mean(x, dim=1, keepdim=True)  # (b, 1, H, W)

        # Fourier域滤波
        def Fx(x):
            return torch.fft.fft2(torch.fft.fftshift(x, dim=(-2, -1)), dim=(-2, -1))

        def FiltX(H, x):
            return torch.real(torch.fft.ifftshift(torch.fft.ifft2(H * Fx(x), dim=(-2, -1)), dim=(-2, -1)))

        # 计算 Hs 和 Hs_conj
        Hs = Fx(self.psf)  # PSF 的频谱
        Hs_conj = torch.conj(Hs)  # 共轭
        HtH = torch.abs(Hs * Hs_conj)  # 幅值平方

        # 重建图像
        xFilt_mult = 1 / (HtH + self.gamm_l2)

        # 进行重建
        numerator = FiltX(Hs_conj, x)
        R_nxt = FiltX(xFilt_mult, numerator)
        # R_nxt = R_nxt * self.normalizer

        # 进行裁剪
        x_start, y_start, crop_width, crop_height = self.crop_area
        R_nxt = R_nxt[:, :, y_start:y_start + crop_height, x_start:x_start + crop_width]

        # 调整大小到 (512, 512)并归一化
        R_nxt = F.interpolate(R_nxt, size=(512, 512), mode='bilinear', align_corners=False)
        # 计算最小值和最大值
        min_val = R_nxt.min()
        max_val = R_nxt.max()
        # 归一化到 [0, 1] 范围
        R_nxt = (R_nxt - min_val) / (max_val - min_val)
        R_nxt = R_nxt * self.normalizer

        # 翻转
        R_nxt = torch.flip(R_nxt, dims=[-1])  # 沿着最后一个维度（宽度）进行水平翻转
        R_nxt = torch.flip(R_nxt, dims=[-2])  # 沿着倒数第二个维度（高度）进行垂直翻转

        # 将单通道结果堆叠为三通道
        R_nxt = R_nxt.repeat(1, 3, 1, 1)  # (b, 1, H, W) -> (b, 3, H, W)

        return R_nxt


# 加载图像并转换为张量
def load_image(image_path, size=(1080, 1440)):
    image = Image.open(image_path).convert('L')  # 转为灰度图
    transform = transforms.Compose([
        transforms.Resize(size),  # 调整大小
        transforms.ToTensor(),  # 转为张量
    ])
    img_tensor = transform(image).unsqueeze(0)  # 增加 batch 维度
    return img_tensor

if __name__ == '__main__':
    # 设置路径
    image_paths = [
        r"E:\cqy\phase_data\0116\measure\image (1).png",
        r"E:\cqy\phase_data\0116\measure\image (2).png",
    ]
    # 加载输入图像
    images = [load_image(path) for path in image_paths]
    b = torch.cat(images, dim=0).cuda()  # 合并为一个批次 (b, 1, 1080, 1440)

    psf_path = r"D:\cqy\phase_data\0116\psf\psf.png"
    # b = torch.randn(2,3,1080,1440).cuda()
    # 实例化模型
    model = FFTlayer(psf_image_path=psf_path).cuda()

    # 模型前向推理
    output = model(b)

    # 打印输出的形状
    print("Output shape: ", output.shape) #(b,3,512,512)


    # 保存结果为图像
    # def save_image(tensor, filename):
    #     tensor = tensor.squeeze(0)  # 移除 batch 维度
    #     tensor = tensor.cpu().detach()  # 移动到 CPU 并断开计算图
    #     tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # 归一化到 [0, 1]
    #     img_pil = transforms.ToPILImage()(tensor)  # 转为 PIL 图像
    #     img_pil.save(filename)
    #
    #
    # # 保存重建的图像
    # save_image(output[0], r"E:\cqy\phase_data\0116\reconstructed_image_1.png")
    # save_image(output[1], r"E:\cqy\phase_data\0116\reconstructed_image_2.png")

    # print("Images saved successfully.")
