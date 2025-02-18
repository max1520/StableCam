import os
from PIL import Image
from torchvision import transforms
import torch
from FFTlayer import FFTlayer

# 定义设备
device = torch.device('cuda')
psf_path = r"D:\cqy\phase_data\0116\psf\psf.png"

# 定义模型
trainablecamerainversion =model = FFTlayer(psf_image_path=psf_path).cuda()

# 定义转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量并归一化到 [0, 1]
])

# 读取文件夹中的所有图像
input_dir = r"D:\cqy\phase_data\0116\eval\measure"
output_dir = r"D:\cqy\stableSR\CFW_data\eval_data\phase_calibration\WY"

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 遍历目录中的所有图像文件
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):  # 只处理PNG文件，可以根据需要修改
        image_path = os.path.join(input_dir, filename)
        print(f"正在处理图像: {filename}")

        # 读取图像
        image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式

        # 应用转换
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 通过模型生成图像
        y = trainablecamerainversion(image_tensor)
        print(f"输出图像形状: {y.shape}")  # 形状应该是 (b, 3, 512, 512)

        y = y.squeeze(0)  # 形状变为 (3, 512, 512)

        # 转换为PIL图像
        to_pil = transforms.ToPILImage()
        output_image = to_pil(y)

        # 构建保存路径
        save_path = os.path.join(output_dir, f"{filename}")

        # 保存图像
        output_image.save(save_path)
        print(f"图像已保存到: {save_path}")
