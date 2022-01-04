from loss import *
import data
from argparse import ArgumentParser
from models.backbone import *
import torch
import torch.nn as nn
from utils import AverageMeter
import time
import sys
import timm
from timm.models import model_parameters
import torch.backends.cudnn as cudnn

import io
from PIL import Image
import requests
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

img = Image.open("test.jpg")

gt_size = [640,480]
output_size = (480, 640)
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]

transform_pipeline = transforms.Compose([
    transforms.Resize((gt_size[1], gt_size[0])),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])    

img = transform_pipeline(img)

img = img.unsqueeze(0)
img = Variable(img).cuda()

def model_load_state_dict(student , teacher, path_state_dict):
    student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
    teacher.load_state_dict(torch.load(path_state_dict)["teacher"], strict=True)
    print("loaded pre-trained student and teacher")

teacher = OFA595(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout="simple")
student = EEEAC2(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout="simple")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher.to(device)
student.to(device)

model_load_state_dict(student , teacher, "pre-trained/model_ofa1k.pt")

teacher.eval()
student.eval()

prediction_t = teacher(img)
prediction_s = student(img)

from torchvision.utils import save_image

save_image(prediction_t, 'teacher.png')
save_image(prediction_s, 'student.png')

print(prediction_t.size())

#################### visual color ##############

import cv2
#input
background = cv2.imread('test.jpg')
#saliency prediction
saliency = cv2.imread('teacher.png', 0)
heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_HOT)

added_image = cv2.addWeighted(background,0.8, heatmap, 0.8, 0)

cv2.imwrite('combined.png', added_image)

#cv2.imshow('heatmap', added_image)
#cv2.waitKey()
