from models.backbone import *
from torch.autograd import Variable
import numpy as np
import torch
import time
import statistics

def computeTime(model, device='cpu', size=384):
    inputs = torch.randn(1, 3, size, size)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < 10:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append((time.time() - start_time)*1000)
        i += 1
    print('Avg execution time (ms): {:.3f}'.format(statistics.mean(time_spent)))
    print('SD execution time (ms): {:.3f}'.format(statistics.stdev(time_spent)))


output_size = (480, 640)
readout = "simple"


student = EEEAC2(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout=readout)
# student = EEEAC1(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout=readout)
# student = MobileNetV2(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout=readout)
# student = MobileNetV3_1k(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout=readout)
# student = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout=readout)
# student = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout=readout)
# student = EfficientNetB7(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout=readout)
# student = GhostNet(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout=readout)
# student = ResT(num_channels=3, train_enc=True, load_weight=1, output_size=output_size, readout=readout)

computeTime(student)

# x_image = Variable(torch.randn(1, 3, 384, 384))
# y = model(x_image)

