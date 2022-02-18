from collections import OrderedDict
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import darmo
from models.readout import ReadOut
from models.head import Decoder
import timm

def _remove_module(state_dict, index=9):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[index:]
        new_state_dict[name] = v
    return new_state_dict

class PNASModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640)):
        super(PNASModel, self).__init__()

        self.output_size = output_size

        self.pnas = darmo.create_model("pnas5", pretrained=True)

        for param in self.pnas.parameters():
            param.requires_grad = train_enc

        self.padding = nn.ConstantPad2d((0,1,0,1),0)
        self.drop_path_prob = 0

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 4320, out_channels = 512, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 512+2160, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1080+256, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 540, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
        
        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))
    
    def forward(self, images):
        batch_size = images.size(0)

        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==11:
                out5 = s1
        
        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)
        
        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        
        x = torch.cat((x,out1), 1)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)

        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)

        if not self.training:
            x = self.blur(x)

        x = x.squeeze(1)
        return x

class DenseModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(DenseModel, self).__init__()

        self.dense = models.densenet161(pretrained=bool(load_weight)).features

        for param in self.dense.parameters():
            param.requires_grad = train_enc

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_layer0 = nn.Sequential(*list(self.dense)[:3])
        self.conv_layer1 = nn.Sequential(
        	self.dense.pool0,
        	self.dense.denseblock1,
        	*list(self.dense.transition1)[:3]
        )
        self.conv_layer2 = nn.Sequential(
        	self.dense.transition1[3],
        	self.dense.denseblock2,
        	*list(self.dense.transition2)[:3]
        )
        self.conv_layer3 = nn.Sequential(
        	self.dense.transition2[3],
        	self.dense.denseblock3,
        	*list(self.dense.transition3)[:3]
        )
        self.conv_layer4 = nn.Sequential(
        	self.dense.transition3[3],
        	self.dense.denseblock4
        )

        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 2208, out_channels = 512, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 512+1056, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 384+256, out_channels = 192, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 192+192, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 96+96, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))

    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer0(images)
        out2 = self.conv_layer1(out1)
        out3 = self.conv_layer2(out2)
        out4 = self.conv_layer3(out3)
        out5 = self.conv_layer4(out4)

        # assert out1.size() == (batch_size, 96, 128, 128)
        # assert out2.size() == (batch_size, 192, 64, 64)
        # assert out3.size() == (batch_size, 384, 32, 32)
        # assert out4.size() == (batch_size, 1056, 16, 16)
        # assert out5.size() == (batch_size, 2208, 8, 8)

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        
        x = torch.cat((x,out1), 1)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)

        x = F.interpolate(x, size=(480,640), mode='bilinear', align_corners=False)

        if not self.training:
            x = self.blur(x)

        x = x.squeeze(1)
        return x

class ResNetModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, pretrained='deeplab_coco'):
        super(ResNetModel, self).__init__()

        self.num_channels = num_channels
        self.resnet = models.resnet50(pretrained=bool(load_weight))
        # delete fc layer from ResNet
        del self.resnet.fc.weight
        del self.resnet.fc.bias

        if pretrained == '21k':
            init = torch.load("resnet50_miil_21k.pth", map_location='cpu')['state_dict']
            del init['fc.weight']
            del init['fc.bias']
        elif pretrained == 'places365':
            init = torch.load("resnet50_places365.pth.tar", map_location='cpu')['state_dict']
            init = _remove_module(init, 7)
            del init['fc.weight']
            del init['fc.bias']
        elif pretrained == 'fcn_coco':
            init = torch.load("fcn_resnet50_coco-1167a1af.pth", map_location='cpu')
            remove = []
            for data in init.keys():
                if 'classifier.' in data:
                    remove.append(data)
            for key in remove:
                del init[key]
            init = _remove_module(init)
        elif pretrained == 'deeplab_coco': 
            init = torch.load("deeplabv3_resnet50_coco-cd0a2569.pth", map_location='cpu')
            remove = []
            for data in init.keys():
                if 'classifier.' in data:
                    remove.append(data)
            for key in remove:
                del init[key]
            init = _remove_module(init)
        # Load state dict to ResNet Model
        self.resnet.load_state_dict(init, strict=True)
        
        for param in self.resnet.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.conv_layer2 = nn.Sequential(
            self.resnet.maxpool,
            self.resnet.layer1
        )
        self.conv_layer3 = self.resnet.layer2
        self.conv_layer4 = self.resnet.layer3
        self.conv_layer5 = self.resnet.layer4

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))
    
    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.deconv_layer0(out5)
        assert out5.size() == (batch_size, 1024, 16, 16)

        x = torch.cat((out5,out4), 1)
        assert x.size() == (batch_size, 2048, 16, 16)
        x = self.deconv_layer1(x)
        assert x.size() == (batch_size, 512, 32, 32)
        
        x = torch.cat((x, out3), 1)
        assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        assert x.size() == (batch_size, 64, 128, 128)
        
        x = torch.cat((x, out1), 1)
        assert x.size() == (batch_size, 128, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        assert x.size() == (batch_size, 1, 256, 256)
        if not self.training:
            x = self.blur(x)
        x = x.squeeze(1)
        assert x.size() == (batch_size, 256, 256)
        return x

class VGGModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640)):
        super(VGGModel, self).__init__()

        self.output_size = output_size
        self.num_channels = num_channels
        self.vgg = models.vgg16(pretrained=bool(load_weight)).features
        
        for param in self.vgg.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = self.vgg[:7]
        self.conv_layer2 = self.vgg[7:12]
        self.conv_layer3 = self.vgg[12:19]
        self.conv_layer4 = self.vgg[19:24]
        self.conv_layer5 = self.vgg[24:]

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))
    
    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.linear_upsampling(out5)
        #assert out5.size() == (batch_size, 512, 16, 16)

        x = torch.cat((out5,out4), 1)
        #assert x.size() == (batch_size, 1024, 16, 16)
        x = self.deconv_layer1(x)
        #assert x.size() == (batch_size, 512, 32, 32)
        
        x = torch.cat((x, out3), 1)
        #assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        #assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        #assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        #assert x.size() == (batch_size, 128, 128, 128)
        
        x = torch.cat((x, out1), 1)
        #assert x.size() == (batch_size, 256, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)

        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)

        if not self.training:
            x = self.blur(x)

        x = x.squeeze(1)
        return x


class ResNestModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, model='resnest50', output_size=(480,640)):
        super(ResNestModel, self).__init__()
        self.output_size = output_size
        self.num_channels = num_channels
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=False)
        self.resnet = torch.hub.load('zhanghang1989/ResNeSt', model, pretrained=True)
        
        for param in self.resnet.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.conv_layer2 = nn.Sequential(
            self.resnet.maxpool,
            self.resnet.layer1
        )
        self.conv_layer3 = self.resnet.layer2
        self.conv_layer4 = self.resnet.layer3
        self.conv_layer5 = self.resnet.layer4

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv_layer0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 2048, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 512, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            #Last2(64, 1),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            #nn.Conv2d(in_channels = 1*5, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))
    
    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.deconv_layer0(out5)
        #assert out5.size() == (batch_size, 1024, 16, 16)

        x = torch.cat((out5,out4), 1)
        #assert x.size() == (batch_size, 2048, 16, 16)
        x = self.deconv_layer1(x)
        #assert x.size() == (batch_size, 512, 32, 32)
        
        x = torch.cat((x, out3), 1)
        #assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        #assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        #assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        #assert x.size() == (batch_size, 64, 128, 128)
        
        x = torch.cat((x, out1), 1)
        #assert x.size() == (batch_size, 128, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        #assert x.size() == (batch_size, 1, 256, 256)

        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        if not self.training:
            x = self.blur(x)

        x = x.squeeze(1)
        #assert x.size() == (batch_size, 256, 256)
        return x


class ResNetModelCustom(nn.Module):
    def __init__(self, num_channels=3, train_enc=False, load_weight=1, pretrained='21k'):
        super(ResNetModelCustom, self).__init__()

        self.num_channels = num_channels
        #self.resnet = models.resnet50(pretrained=bool(load_weight))
        self.model = darmo.create_model("resnet50_21k", num_classes=1000, pretrained=True)

        # if pretrained == '21k':
        #     init = torch.load("resnet50_miil_21k.pth", map_location='cpu')['state_dict']
        #     del init['fc.weight']
        #     del init['fc.bias']
        #     # Load state dict to ResNet Model
        #     self.resnet.load_state_dict(init, strict=True)
        # elif pretrained == 'places365':
        #     init = torch.load("resnet50_places365.pth.tar", map_location='cpu')['state_dict']
        #     init = _remove_module(init, 7)
        #     del init['fc.weight']
        #     del init['fc.bias']
        #     # Load state dict to ResNet Model
        #     self.resnet.load_state_dict(init, strict=True)
        # elif pretrained == 'fcn_coco':
        #     init = torch.load("fcn_resnet50_coco-1167a1af.pth", map_location='cpu')
        #     remove = []
        #     for data in init.keys():
        #         if 'classifier.' in data:
        #             remove.append(data)
        #     for key in remove:
        #         del init[key]
        #     init = _remove_module(init)
        #     # Load state dict to ResNet Model
        #     self.resnet.load_state_dict(init, strict=True)
        # elif pretrained == 'deeplab_coco': 
        #     init = torch.load("deeplabv3_resnet50_coco-cd0a2569.pth", map_location='cpu')
        #     remove = []
        #     for data in init.keys():
        #         if 'classifier.' in data:
        #             remove.append(data)
        #     for key in remove:
        #         del init[key]
        #     init = _remove_module(init)
        #     # Load state dict to ResNet Model
        #     self.resnet.load_state_dict(init, strict=True)

        print("Loaded pre-trained: ", pretrained)
        
        for param in self.model.resnet.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = nn.Sequential(
            self.model.resnet.conv1,
            self.model.resnet.bn1,
            self.model.resnet.relu
        )
        self.conv_layer2 = nn.Sequential(
            self.model.resnet.maxpool,
            self.model.resnet.layer1
        )
        self.conv_layer3 = self.model.resnet.layer2
        self.conv_layer4 = self.model.resnet.layer3
        self.conv_layer5 = self.model.resnet.layer4

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))
    
    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.deconv_layer0(out5)
        #assert out5.size() == (batch_size, 1024, 16, 16)

        x = torch.cat((out5,out4), 1)
        #assert x.size() == (batch_size, 2048, 16, 16)
        x = self.deconv_layer1(x)
        #assert x.size() == (batch_size, 512, 32, 32)
        
        x = torch.cat((x, out3), 1)
        #assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        #assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        #assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        #assert x.size() == (batch_size, 64, 128, 128)
        
        x = torch.cat((x, out1), 1)
        #assert x.size() == (batch_size, 128, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        #assert x.size() == (batch_size, 1, 256, 256)

        x = F.interpolate(x, size=(480,640), mode='bilinear', align_corners=False)

        if not self.training:
            x = self.blur(x)
            
        x = x.squeeze(1)
        return x

class MobileNetV2(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(MobileNetV2, self).__init__()

        self.mobilenet = torch.hub.load('pytorch/vision:v0.4.0', 'mobilenet_v2', pretrained=True).features

        # for param in self.mobilenet.parameters():
        #     param.requires_grad = train_enc

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_layer1 = self.mobilenet[:2]
        self.conv_layer2 = self.mobilenet[2:4]
        self.conv_layer3 = self.mobilenet[4:7]
        self.conv_layer4 = self.mobilenet[7:14]
        self.conv_layer5 = self.mobilenet[14:]

        self.head = Decoder(channels=[1280, 96, 32, 24, 16, 16], act="ReLU", output_size=output_size, readout=readout) 

    def forward(self, images):
        batch_size = images.size(0)
        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        x = self.head(out1, out2, out3, out4, out5)
        return x

class EEEAC2(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(EEEAC2, self).__init__()

        self.num_channels = num_channels
        self.eeeac2 =  darmo.create_model("eeea_c2", num_classes=1000, pretrained=True, auxiliary=True)

        for param in self.eeeac2.parameters():
            param.requires_grad = train_enc

        del self.eeeac2.feature_mix_layer 
        del self.eeeac2.classifier 

        self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="Swish", output_size=output_size, readout=readout) 

    def forward(self, x):
        x = self.eeeac2.first_conv(x)
        out1 = x
        for i in range(len(self.eeeac2.blocks)):
            x = self.eeeac2.blocks[i](x)
            if i == 3:
                out2 = x
            elif i == 6:
                out3 = x
            elif i == 14:
                out4 = x

        x = self.eeeac2.final_expand_layer(x)
        out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class EEEAC1(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(EEEAC1, self).__init__()

        self.num_channels = num_channels
        self.model =  darmo.create_model("eeea_c1", num_classes=1000, pretrained=True, auxiliary=True)

        # for param in self.model.parameters():
        #     param.requires_grad = train_enc

        del self.model.feature_mix_layer 
        del self.model.classifier  

        self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="Swish", output_size=output_size, readout=readout)

    def forward(self, x):
        x = self.model.first_conv(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)
            if i == 2:
                out2 = x
            elif i == 4:
                out3 = x
            elif i == 8:
                out4 = x
        x = self.model.final_expand_layer(x)
        out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class GhostNet(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(GhostNet, self).__init__()

        self.model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
        
        del self.model.global_pool
        del self.model.conv_head
        del self.model.act2
        del self.model.classifier
        
        self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size, readout=readout) 

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)
            if i == 2:
                out2 = x
            elif i == 4:
                out3 = x
            elif i == 6:
                out4 = x
        out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class MobileNetV3(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(MobileNetV3, self).__init__()

        self.model = models.mobilenet_v3_large(pretrained=True).features

        self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size, readout=readout) 

    def forward(self, x):
        for i in range(len(self.model)):
            x = self.model[i](x)      
            if i == 0:
                out1 = x
            elif i == 3:
                out2 = x
            elif i == 6:
                out3 = x
            elif i == 12:
                out4 = x
        out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class EfficientNet(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(EfficientNet, self).__init__()

        self.model = timm.create_model('tf_efficientnet_b0', pretrained=True) 

        del self.model.global_pool
        del self.model.classifier

        self.head = Decoder(channels=[1280, 112, 40, 24, 32, 32], act="ReLU", output_size=output_size, readout=readout) 

    def forward(self, x):
        x = self.model.conv_stem(x)  
        x = self.model.bn1(x)  
        x = self.model.act1(x)  
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)            
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
        x = self.model.conv_head(x)  
        x = self.model.bn2(x)  
        x = self.model.act2(x)  
        out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class EfficientNetB4(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(EfficientNetB4, self).__init__()

        self.model = timm.create_model('tf_efficientnet_b4', pretrained=True) 

        del self.model.global_pool
        del self.model.classifier

        self.head = Decoder(channels=[1792, 160, 56, 32, 48, 48], act="ReLU", output_size=output_size, readout=readout) 

    def forward(self, x):
        x = self.model.conv_stem(x)  
        x = self.model.bn1(x)  
        x = self.model.act1(x)  
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)            
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
        x = self.model.conv_head(x)  
        x = self.model.bn2(x)  
        x = self.model.act2(x)  
        out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class EfficientNetB7(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(EfficientNetB7, self).__init__()

        self.model = timm.create_model('tf_efficientnet_b7', pretrained=True) 

        del self.model.global_pool
        del self.model.classifier

        self.head = Decoder(channels=[2560, 224, 80, 48, 64, 64], act="ReLU", output_size=output_size, readout=readout) 

    def forward(self, x):
        x = self.model.conv_stem(x)  
        x = self.model.bn1(x)  
        x = self.model.act1(x)  
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)            
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
        x = self.model.conv_head(x)  
        x = self.model.bn2(x)  
        x = self.model.act2(x)  
        out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class tresnet(nn.Module):
    def __init__(self, num_channels=3, train_enc=False, load_weight=1, pretrained='21k', output_size=(480,640)):
        super(tresnet, self).__init__()
        self.output_size = output_size
        self.num_channels = num_channels
        if pretrained == '21k':
            self.resnet = timm.create_model('tresnet_m_miil_in21k', pretrained=True) 
        else: 
            self.resnet = timm.create_model('tresnet_m', pretrained=True) 
        print("Loaded pre-trained: ", pretrained)
        
        for param in self.resnet.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = nn.Sequential(
            self.resnet.body.SpaceToDepth,
            self.resnet.body.conv1,
        )

        self.conv_layer2 = self.resnet.body.layer1
        
        self.conv_layer3 = self.resnet.body.layer2
        self.conv_layer4 = self.resnet.body.layer3
        self.conv_layer5 = self.resnet.body.layer4

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 2048, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=1)
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))
    
    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        #print(out1.size())
        out2 = self.conv_layer2(out1)
        #print(out2.size())
        out3 = self.conv_layer3(out2)
        #print(out3.size())
        out4 = self.conv_layer4(out3)
        #print(out4.size())
        out5 = self.conv_layer5(out4)
        #print(out5.size())

        out5 = self.deconv_layer0(out5)
        #assert out5.size() == (batch_size, 1024, 16, 16)

        x = torch.cat((out5,out4), 1)
        #assert x.size() == (batch_size, 2048, 16, 16)
        x = self.deconv_layer1(x)
        #assert x.size() == (batch_size, 512, 32, 32)
        
        x = torch.cat((x, out3), 1)
        #assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        #assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        #assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        #assert x.size() == (batch_size, 64, 128, 128)
        x = torch.cat((x, out1), 1)
        #assert x.size() == (batch_size, 128, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        #assert x.size() == (batch_size, 1, 256, 256)

        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)

        if not self.training:
            x = self.blur(x)
            
        x = x.squeeze(1)
        return x

class MobileNetV3_21k(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(MobileNetV3_21k, self).__init__()

        self.model = timm.create_model('mobilenetv3_large_100_miil_in21k', pretrained=True)

        del self.model.global_pool
        del self.model.conv_head
        del self.model.act2
        del self.model.flatten
        del self.model.classifier

        self.stem = nn.Sequential(
            self.model.conv_stem,
            self.model.bn1,
            self.model.act1,
        )

        self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size, readout=readout) 

    def forward(self, x):
        x = self.stem(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x) 
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
            elif i == 6:
                out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class MobileNetV3_1k(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(MobileNetV3_1k, self).__init__()

        self.model = timm.create_model('mobilenetv3_large_100_miil', pretrained=True)

        del self.model.global_pool
        del self.model.conv_head
        del self.model.act2
        del self.model.flatten
        del self.model.classifier

        self.stem = nn.Sequential(
            self.model.conv_stem,
            self.model.bn1,
            self.model.act1,
        )

        self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size, readout=readout) 

    def forward(self, x):
        x = self.stem(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x) 
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
            elif i == 6:
                out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class OFA595(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(OFA595, self).__init__()

        self.num_channels = num_channels
        self.model =  darmo.create_model("ofa595_1k", num_classes=1000, pretrained=True, auxiliary=True)

        for param in self.model.parameters():
            param.requires_grad = train_enc

        del self.model.feature_mix_layer 
        del self.model.classifier 

        self.head = Decoder(channels=[1152, 136, 48, 32, 24, 24], act="Swish", output_size=output_size, readout=readout) 

    def forward(self, x):
        x = self.model.first_conv(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)
            if i == 3:
                out2 = x
            elif i == 7:
                out3 = x
            elif i == 15:
                out4 = x

        x = self.model.final_expand_layer(x)
        out5 = x
        x = self.head(out1, out2, out3, out4, out5)
        return x

class ResNetModel1k(nn.Module):
    def __init__(self, num_channels=3, train_enc=False, load_weight=1, pretrained='1k', output_size=(480,640)):
        super(ResNetModel1k, self).__init__()
        self.output_size = output_size
        self.num_channels = num_channels
        self.model = models.resnet50(pretrained=bool(load_weight))

        print("Loaded pre-trained: ", pretrained)
        
        for param in self.model.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu
        )
        self.conv_layer2 = nn.Sequential(
            self.model.maxpool,
            self.model.layer1
        )
        self.conv_layer3 = self.model.layer2
        self.conv_layer4 = self.model.layer3
        self.conv_layer5 = self.model.layer4

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))
    
    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)

        x = self.deconv_layer1(x)

        x = torch.cat((x, out3), 1)

        x = self.deconv_layer2(x)

        x = torch.cat((x, out2), 1)

        x = self.deconv_layer3(x)

        x = torch.cat((x, out1), 1)

        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)

        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)

        if not self.training:
            x = self.blur(x)
            
        x = x.squeeze(1)
        return x


class ResT(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple"):
        super(ResT, self).__init__()

        self.model = darmo.create_model("rest_large", num_classes=1000, pretrained=True,)

        del self.model.avg_pool
        del self.model.head

        lite  = [512, 256, 128, 64, 64, 64]
        large = [768, 384, 192, 96, 96, 96]
        self.head = Decoder(channels=large, act="ReLU", 
                    output_size=output_size, readout=readout,
                    upsampling=[True, True, True, False, True, True]) 

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.model.stem(x)
        out1 = x
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # stage 1
        for blk in self.model.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        out2 = x

        # stage 2
        x, (H, W) = self.model.patch_embed_2(x)
        for blk in self.model.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        out3 = x

        # stage 3
        x, (H, W) = self.model.patch_embed_3(x)
        for blk in self.model.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        out4 = x

        # stage 4
        x, (H, W) = self.model.patch_embed_4(x)
        for blk in self.model.stage4:
            x = blk(x, H, W)
        x = self.model.norm(x)

        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        out5 = x

        x = self.head(out1, out2, out3, out4, out5)
        return x