import torch
import torch.nn as nn
import torch.nn.functional as F

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, size=1, act="ReLU"):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(size),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True) if act is "ReLU" else MemoryEfficientSwish(),
            )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ReadOut(nn.Module):
    def __init__(self, input, output, act="ReLU"):
        super(ReadOut, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=input, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True) if act is "ReLU" else MemoryEfficientSwish(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=input, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels=input, out_channels=input, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True) if act is "ReLU" else MemoryEfficientSwish(),
        )
        self.branch3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input, out_channels=input, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True) if act is "ReLU" else MemoryEfficientSwish(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(input, input, (1,7), stride=(1, 1), padding=(0, 3), bias=False),
            nn.Conv2d(input, input, (7,1), stride=(1, 1), padding=(3, 0), bias=False),
            nn.ReLU(inplace=True) if act is "ReLU" else MemoryEfficientSwish(),
        )
        self.branch5 = nn.Sequential(
            ASPPPooling(input, input, size=1, act=act)
        )
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=input*5, out_channels=output*5, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True) if act is "ReLU" else MemoryEfficientSwish(),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = F.adaptive_avg_pool2d(x, (h//2, w//2))
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        y = torch.cat([x1, x2, x3, x4, x5], 1)
        y = self.last(y)
        y = F.interpolate(y, (h, w), mode='bilinear', align_corners=False)
        return y