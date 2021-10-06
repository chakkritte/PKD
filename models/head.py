import torch.nn as nn
import torch
import torch.nn.functional as F
import kornia
from models.readout import ReadOut, MemoryEfficientSwish

def _make_output(input, output, act="ReLU", upsampling=True):
    linear_upsampling = nn.UpsamplingNearest2d(scale_factor=2)
    if upsampling:
        return nn.Sequential(
            nn.Conv2d(input, out_channels = output, kernel_size = 3, padding = 1, bias = False),
            nn.ReLU(inplace=True) if act is "ReLU" else MemoryEfficientSwish(),
            linear_upsampling,
        )
    else:
        if act is not 'Sigmoid':
            return nn.Sequential(
                nn.Conv2d(input, out_channels = output, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace=True) if act is "ReLU" else MemoryEfficientSwish(),
            )
        elif act is 'Sigmoid': 
            return nn.Sequential(
                nn.Conv2d(input, out_channels = output, kernel_size = 3, padding = 1, bias = False),
                nn.Sigmoid(),
            )

class Decoder(nn.Module):
    def __init__(self, channels=[960, 112, 40, 24, 16, 16], act="Swish", 
                output_size=(480,640), readout="simple", 
                upsampling=[True, True, True, True, True, True]):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.deconv_layer0 = _make_output(input=channels[0], output=channels[1], act=act, upsampling=upsampling[0])
        self.deconv_layer1 = _make_output(input=channels[1]+channels[1], output=channels[2], act=act, upsampling=upsampling[1])
        self.deconv_layer2 = _make_output(input=channels[2]+channels[2], output=channels[3], act=act, upsampling=upsampling[2])
        self.deconv_layer3 = _make_output(input=channels[3]+channels[3], output=channels[4], act=act, upsampling=upsampling[3])
        self.deconv_layer4 = _make_output(input=channels[4]+channels[4], output=channels[5], act=act, upsampling=upsampling[4])
        if readout == "simple":
            self.deconv_layer5 = _make_output(input=channels[5], output=channels[5], act=act, upsampling=upsampling[5])
            output_number = channels[5]
        else: 
            self.deconv_layer5 = ReadOut(input=channels[5], output=1, act=act)
            output_number = 5
        self.deconv_layer6 = _make_output(input=output_number, output=1, act="Sigmoid", upsampling=False)
        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))

    def forward(self, out1, out2, out3, out4, out5):
        x0 = self.deconv_layer0(out5)
        
        x1 = torch.cat((x0,out4), 1)
        x1 = self.deconv_layer1(x1)

        x2 = torch.cat((x1, out3), 1)
        x2 = self.deconv_layer2(x2)

        x3 = torch.cat((x2, out2), 1)
        x3 = self.deconv_layer3(x3)

        x4 = torch.cat((x3, out1), 1)
        x4 = self.deconv_layer4(x4)

        x = self.deconv_layer5(x4)
        x = self.deconv_layer6(x)

        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        if not self.training:
            x = self.blur(x)
        x = x.squeeze(1)
        return x