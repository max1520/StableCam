import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio

class TrainableCameraInversion(nn.Module):
    def __init__(self, initial_mode=None, image_size=None, data_code='1217'):
        super().__init__()

        # 初始化模式和参数
        self.initial_mode = initial_mode
        self.scale = 512 // image_size

        if self.initial_mode not in ['Tikhonov', 'calibration', 'random']:
            raise ValueError(
                f"Expected mode to be 'Tikhonov', 'calibration', or 'random', but got '{self.initial_mode}'")

        if self.initial_mode == 'calibration':
            # 加载 Phi 矩阵
            Phil = sio.loadmat(f"D:/cqy/flat_data/{data_code}/initial_matrix/{image_size}/Phi_rec_left.mat")
            Phir = sio.loadmat(f"D:/cqy/flat_data/{data_code}/initial_matrix/{image_size}/Phi_rec_right.mat")

            # 获取矩阵大小
            self.height_left, self.width_left = Phil['Phi_rec_left'].shape
            self.height_right, self.width_right = Phir['Phi_rec_right'].shape
            print(f"Left matrix size: Height = {self.height_left}, Width = {self.width_left}")
            print(f"Right matrix size: Height = {self.height_right}, Width = {self.width_right}")

            # 将矩阵转换为 nn.Parameter，方便在训练时更新
            self.PhiL = nn.Parameter(torch.from_numpy(Phil['Phi_rec_left']).float().to('cuda'))
            self.PhiR = nn.Parameter(torch.from_numpy(Phir['Phi_rec_right']).float().to('cuda'))

        if self.initial_mode == 'random':
            # 加载 Phi 矩阵
            Phil = sio.loadmat(f"D:/cqy/flat_data/{data_code}/initial_matrix/toplize_{image_size}/Phi_rec_left.mat")
            Phir = sio.loadmat(f"D:/cqy/flat_data/{data_code}/initial_matrix/toplize_{image_size}/Phi_rec_right.mat")

            # 获取矩阵大小
            self.height_left, self.width_left = Phil['Phi_rec_left'].shape
            self.height_right, self.width_right = Phir['Phi_rec_right'].shape
            print(f"Left matrix size: Height = {self.height_left}, Width = {self.width_left}")
            print(f"Right matrix size: Height = {self.height_right}, Width = {self.width_right}")

            # 将矩阵转换为 nn.Parameter，方便在训练时更新
            self.PhiL = nn.Parameter(torch.from_numpy(Phil['Phi_rec_left']).float().to('cuda'))
            self.PhiR = nn.Parameter(torch.from_numpy(Phir['Phi_rec_right']).float().to('cuda'))

    def forward(self, measure):
        # 如果模式是 'calibration'，则执行计算
        if self.initial_mode == 'calibration' or self.initial_mode == 'random':
            # 处理第一个通道
            measure_channel_0 = measure[:, 0, :, :]

            # 进行矩阵乘法
            measure_transformed = torch.matmul(measure_channel_0, self.PhiR[:, :]).permute(0, 2, 1)
            measure_transformed = torch.matmul(measure_transformed, self.PhiL[:, :]).permute(0, 2, 1)

            # 在通道维度上增加维度并重复
            measure_expanded = measure_transformed.unsqueeze(1).repeat(1, 3, 1, 1)

            # 应用 ReLU 激活
            measure = F.relu(measure_expanded)
        else:
            raise ValueError("Expected error in forward")

        if self.scale != 1:
            # 使用双立方插值将尺寸调整到 (512, 512)
            measure = F.interpolate(measure, size=(512, 512), mode='bicubic', align_corners=False)

        return measure

if __name__ == '__main__':
    device = torch.device('cuda')
    trainablecamerainversion = TrainableCameraInversion(initial_mode='random', image_size=128, data_code='0104').to(device)
    x = torch.rand(1,3,540,720).to(device)
    y = trainablecamerainversion(x)
    print(y.shape)  #(1,3,512,512)