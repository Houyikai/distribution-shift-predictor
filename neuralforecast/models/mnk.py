# | export
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class moving_std(nn.Module):
    """
    Moving standard deviation block for time series data
    """

    def __init__(self, kernel_size, stride):
        super(moving_std, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Calculate the standard deviation using a sliding window
        batch_size, seq_len, num_features = x.size()
        pad_size = min((self.kernel_size - 1) // 2, seq_len - 1)
        padded_x = torch.nn.functional.pad(x.permute(0,2,1), (pad_size, pad_size), mode='reflect').permute(0,2,1)
        std_values = []
       
        for i in range(0, seq_len, self.stride):
            window = padded_x[:, i:i+self.kernel_size, :]
            std = torch.std(window, dim=1, unbiased=False)
            std_values.append(std)
        x = torch.stack(std_values, dim=1)
        return x

class moving_norm(nn.Module):  # x: [Batch, Input length, Channel]
    """
    Series norm block
    """

    def __init__(self, kernel_size, stride:int = 1):
        super(moving_norm, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride)
        self.moving_std = moving_std(kernel_size, stride)

    def forward(self, x):
        x_mean_curve = self.moving_avg(x)
        x_std_curve = self.moving_std(x)
        x_std_curve = torch.clamp(x_std_curve, min=0.001)
        x_norm_curve = (x - x_mean_curve) / x_std_curve
        return x_norm_curve, x_mean_curve, x_std_curve
    
class MovingNormlizationKernel(nn.Module):
    
    def __init__(self, kernel_size:int=25, stride:int = 1):
        super(MovingNormlizationKernel, self).__init__()
        self.normlization = moving_norm(kernel_size, stride)
    
    # [Batch, Input length, Channel]
    def stationarize(self, x):
        x_norm_curve, x_mean_curve, x_std_curve = self.normlization(x)
        return x_norm_curve, x_mean_curve, x_std_curve
    
    def destationarize(self, y_norm_curve, y_mean_curve, y_std_curve):
        y = y_norm_curve * y_std_curve + y_mean_curve
        return y