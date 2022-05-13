from loss import *
import data
from argparse import ArgumentParser
from models.backbone import *
import torch
import torch.nn as nn
from utils import AverageMeter
import time
import sys

import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset_dir", type=str, default="data/")
parser.add_argument('--input_size_h',default=256, type=int)
parser.add_argument('--input_size_w',default=256, type=int)
parser.add_argument('--no_workers',default=16, type=int)
parser.add_argument('--log_interval',default=20, type=int)
parser.add_argument('--pretrained',default="pre-trained/model_ofa1k.pt", type=str)

parser.add_argument('--dataset',default="salicon", type=str)
parser.add_argument('--student',default="eeeac2", type=str)
parser.add_argument('--teacher',default="pnas", type=str)

parser.add_argument('--readout',default="simple", type=str)
parser.add_argument('--output_size', default=(480, 640))

parser.add_argument('--mode',default="kd", type=str)
parser.add_argument('--mixed',default=True, type=bool)
parser.add_argument('--seed',default=3407, type=int)

args = parser.parse_args()

def model_load_state_dict(student , teacher, path_state_dict):
    if args.mode == "kd":
        student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
        teacher.load_state_dict(torch.load(path_state_dict)["teacher"], strict=True)
        print("loaded pre-trained student and teacher")
    else: 
        student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
        print("loaded pre-trained student")

if args.dataset != "salicon":
    args.output_size = (384, 384)

if args.student == "eeeac2":
    student = EEEAC2(num_channels=3, train_enc=True, load_weight=True, output_size=args.output_size, readout=args.readout)
elif args.student == "eeeac1":
    student = EEEAC1(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "mbv2":
    student = MobileNetV2(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "mbv3":
    student = MobileNetV3_1k(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "efb0":
    student = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "efb4":
    student = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "efb7":
    student = EfficientNetB7(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "ghostnet":
    student = GhostNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "rest":
    student = ResT(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "vgg":
    student = VGGModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)

torch.multiprocessing.freeze_support()

print(args.dataset)

if args.dataset == 'salicon':
    args.output_size = [480, 640]
    args.input_size = 384
    train_dataset = data.SaliconDataset(args.dataset_dir, train=True, input_size_h=args.input_size, input_size_w=args.input_size)
    val_dataset = data.SaliconDataset(args.dataset_dir, train=False, input_size_h=args.input_size, input_size_w=args.input_size)

elif args.dataset == 'mit1003':
    args.output_size = [384, 384]
    args.input_size = 384
    train_dataset = data.Mit1003Dataset(args.dataset_dir, train=True, input_size_h=args.input_size, input_size_w=args.input_size)
    val_dataset = data.Mit1003Dataset(args.dataset_dir, train=False, input_size_h=args.input_size, input_size_w=args.input_size)

elif args.dataset == 'cat2000':
    args.output_size = [384, 384]
    args.input_size = 384
    train_dataset = data.CAT2000Dataset(args.dataset_dir, train=True, input_size_h=args.input_size, input_size_w=args.input_size)
    val_dataset = data.CAT2000Dataset(args.dataset_dir, train=False, input_size_h=args.input_size, input_size_w=args.input_size)

elif args.dataset == 'pascals':
    args.output_size = [384, 384]
    args.input_size = 384
    train_dataset = data.PASCALSDataset(args.dataset_dir, train=True, input_size_h=args.input_size, input_size_w=args.input_size)
    val_dataset = data.PASCALSDataset(args.dataset_dir, train=False, input_size_h=args.input_size, input_size_w=args.input_size)

elif args.dataset == 'osie':
    args.output_size = [384, 384]
    args.input_size = 384
    train_dataset = data.OSIEDataset(args.dataset_dir, train=True, input_size_h=args.input_size, input_size_w=args.input_size)
    val_dataset = data.OSIEDataset(args.dataset_dir, train=False, input_size_h=args.input_size, input_size_w=args.input_size)

elif args.dataset == 'dutomron':
    args.output_size = [384, 384]
    args.input_size = 384
    train_dataset = data.DUTOMRONDataset(args.dataset_dir, train=True, input_size_h=args.input_size, input_size_w=args.input_size)
    val_dataset = data.DUTOMRONDataset(args.dataset_dir, train=False, input_size_h=args.input_size, input_size_w=args.input_size)

elif args.dataset == 'fiwi':
    args.output_size = [384, 384]
    args.input_size = 384
    train_dataset = data.FIWIDataset(args.dataset_dir, train=True, input_size_h=args.input_size, input_size_w=args.input_size)
    val_dataset = data.FIWIDataset(args.dataset_dir, train=False, input_size_h=args.input_size, input_size_w=args.input_size)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.no_workers, pin_memory=True)

if args.dataset != "salicon":
    args.output_size = (384, 384)

if args.teacher == "ofa595":
    teacher = OFA595(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "tresnet":
    teacher = tresnet(num_channels=3, train_enc=True, load_weight=1, pretrained='1k', output_size=args.output_size)
elif args.teacher == "mbv3":
    teacher = MobileNetV3_1k(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "efb0":
    teacher = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "efb4":
    teacher = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "efb7":
    teacher = EfficientNetB7(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "pnas":
    teacher = PNASModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
elif args.teacher == "vgg":
    teacher = VGGModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)

def get_pretrained(args):
    if args.student == 'eeeac2' and args.teacher == 'ofa595':
        return "pre-trained/model_ofa1k.pt"
    if args.student == 'eeeac2' and args.teacher == 'efb4':
        return "pre-trained/model_efb4.pt"
    if args.student == 'eeeac2' and args.teacher == 'pnas':
        return "pre-trained/model_pnasnet5_1k.pt"

pretrained_path = get_pretrained(args)

model_load_state_dict(student , teacher, pretrained_path)

from ptflops import get_model_complexity_info

print("Teacher:")
macs, params = get_model_complexity_info(teacher, (3, args.input_size, args.input_size), as_strings=True,
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

print("Student:")
macs, params = get_model_complexity_info(student, (3, args.input_size, args.input_size), as_strings=True,
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    student = nn.DataParallel(student)
    if args.mode == "kd":
        teacher = nn.DataParallel(teacher)

student.to(device)

if args.mode == "kd":
    teacher.to(device)
else: 
    teacher = None

def validate(model, loader, device):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    auc_loss = AverageMeter()
    
    for (img, gt, fixations) in loader:
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        pred_map = model(img)

        cc_loss.update(cc(pred_map, gt))    
        kldiv_loss.update(kldiv(pred_map, gt))    
        nss_loss.update(nss(pred_map, fixations))    
        sim_loss.update(similarity(pred_map, gt))    
        auc_loss.update(auc_judd(pred_map, fixations))    

    print('[val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}, AUC : {:.5f}  time:{:3f} minutes'.format(cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, auc_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()

with torch.no_grad():
    if args.mode == "kd":
        # print("Teacher:")
        # _ = validate(teacher, val_loader, device)
        print("Student:")
        cc_loss = validate(student, val_loader, device)
    else :
        cc_loss = validate(student, val_loader, device)