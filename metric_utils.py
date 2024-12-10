import math

class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.sum_sq = 0  # 新增：用于存储平方和
        self.std = 0  # 新增：用于存储标准差

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # 更新平方和
        self.sum_sq += (val ** 2) * n

        # 计算标准差
        if self.count > 0:
            mean_sq = self.sum_sq / self.count
            self.std = math.sqrt(mean_sq - self.avg ** 2)
        else:
            self.std = 0