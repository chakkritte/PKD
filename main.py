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
from torch.optim.swa_utils import AveragedModel, update_bn, SWALR
from adamp import AdamP
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset_dir", type=str, default="/home/mllab/proj/2021/supernet/data/")
parser.add_argument('--input_size_h',default=256, type=int)
parser.add_argument('--input_size_w',default=256, type=int)
parser.add_argument('--no_workers',default=16, type=int)
parser.add_argument('--no_epochs',default=10, type=int)
parser.add_argument('--log_interval',default=20, type=int)
parser.add_argument('--lr_sched',default=True, type=bool)
parser.add_argument('--model_val_path',default="model.pt", type=str)

parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=True, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--l1',default=False, type=bool)
parser.add_argument('--auc',default=True, type=bool)

parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)
parser.add_argument('--auc_coeff',default=1.0, type=float)

parser.add_argument('--dataset',default="salicon", type=str)
parser.add_argument('--model',default="efb0", type=str)
parser.add_argument('--readout',default="simple", type=str)
parser.add_argument('--output_size', default=(480, 640))

parser.add_argument('--mode',default="kd", type=str)
parser.add_argument('--seed',default=3407, type=int)
args = parser.parse_args()

# np.random.seed(args.seed)
# cudnn.benchmark = False
# cudnn.deterministic = True
# torch.manual_seed(args.seed)
# cudnn.enabled=True
# torch.cuda.manual_seed(args.seed)

def model_load_state_dict(student , teacher, path_state_dict):
    if args.mode == "kd":
        student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
        teacher.load_state_dict(torch.load(path_state_dict)["teacher"], strict=True)
        print("loaded pre-trained student and teacher")
    else: 
        student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
        print("loaded pre-trained student")

def loss_func(pred_map, gt, fixations, args):
    loss = torch.FloatTensor([0.0]).cuda()
    criterion = nn.L1Loss()
    if args.kldiv:
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if args.cc:
        loss += args.cc_coeff * cc(pred_map, gt)
    if args.nss:
        loss += args.nss_coeff * nss(pred_map, fixations)
    if args.l1:
        loss += args.l1_coeff * criterion(pred_map, gt)
    if args.sim:
        loss += args.sim_coeff * similarity(pred_map, gt)
    if args.auc:
        loss += args.auc_coeff * auc_judd(pred_map, fixations)
    return loss

if args.dataset != "salicon":
    args.output_size = (384, 384)

if args.model == "eeeac2":
    student = EEEAC2(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.model == "eeeac1":
    student = EEEAC1(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.model == "mbv2":
    student = MobileNetV2(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.model == "mbv3":
    student = MobileNetV3_1k(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.model == "efb0":
    student = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.model == "efb4":
    student = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.model == "efb7":
    student = EfficientNetB7(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.model == "ghostnet":
    student = GhostNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.model == "rest":
    student = ResT(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)

torch.multiprocessing.freeze_support()

#args.output_size = [args.input_size_h, args.input_size_w]
#args.input_size = 384
print(args.dataset)

if args.dataset == 'salicon':
    args.output_size = [480, 640]
    args.input_size = 384
    train_dataset = data.SaliconDataset(args.dataset_dir, train=True, input_size_h=args.output_size[0], input_size_w=args.output_size[1])
    val_dataset = data.SaliconDataset(args.dataset_dir, train=False, input_size_h=args.output_size[0], input_size_w=args.output_size[1])

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

#teacher = ResNestModel(num_channels=3, train_enc=True, load_weight=1, model='resnest50', output_size=args.output_size)
#teacher = ResNetModelCustom(num_channels=3, train_enc=True, load_weight=1,)
#teacher = tresnet(num_channels=3, train_enc=True, load_weight=1,)
#teacher = MobileNetV3_21k(num_channels=3, train_enc=True, load_weight=1,)
#teacher = OFA595(num_channels=3, train_enc=True, load_weight=1,)

if args.dataset != "salicon":
    args.output_size = (384, 384)

#1k
teacher = OFA595(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
#teacher = tresnet(num_channels=3, train_enc=True, load_weight=1, pretrained='1k', output_size=args.output_size)
#teacher = ResNestModel(num_channels=3, train_enc=True, load_weight=1)
#teacher = MobileNetV3_1k(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
#teacher = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
#teacher = PNASModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
#teacher = VGGModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
#teacher = ResNetModel1k(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
#teacher = DenseModel(num_channels=3, train_enc=True, load_weight=1,)

#teacher = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)

if args.dataset != "salicon":
    model_load_state_dict(student , teacher, "pre-trained/model_ofa1k.pt")

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

if args.mode == "kd":
    params_group = [
        {"params": list(filter(lambda p: p.requires_grad, teacher.parameters())), "lr" : args.learning_rate },
        {"params": list(filter(lambda p: p.requires_grad, student.parameters())), "lr" : args.learning_rate*10 },
    ]
else : 
    params_group = [
        {"params": list(filter(lambda p: p.requires_grad, student.parameters())), "lr" : args.learning_rate*10 },
    ]

optimizer = torch.optim.Adam(params_group)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.no_epochs))

print(device)

def validate(model, loader, epoch, device):
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

    print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}, AUC : {:.5f}  time:{:3f} minutes'.format(epoch, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, auc_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()
    
    return cc_loss.avg + nss_loss.avg + sim_loss.avg + (1-kldiv_loss.avg) + auc_loss.avg

def train(student, optimizer, loader, epoch, device, args, teacher):
    student.train()
    if args.mode == "kd":
        teacher.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    #resize_list = [384, 416, 480]
    #active_resolution = random.choice(range(12, 21)) * 32

    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()

        # multi-sclae training (320-640 pixels) every 10 batches
        #if (idx+1) % 10 == 0:
        #    active_resolution = random.choice(range(12, 21)) * 32

        #img = torch.nn.functional.interpolate(img, size=active_resolution, mode='bicubic', align_corners=False)

        if args.mode == "kd":
            with torch.cuda.amp.autocast():
                pred_map = teacher(img)
                loss = loss_func(pred_map, gt, fixations, args)

            #loss.backward()
            #scaler.scale(loss).backward()

        with torch.cuda.amp.autocast():
            pred_map_student = student(img)
            if args.mode == "kd":
                loss_s = loss_func(pred_map_student, pred_map.detach(), fixations, args)
            else :
                loss_s = loss_func(pred_map_student, gt, fixations, args)

        #loss.backward(retain_graph=True)
        #scaler.scale(loss_s).backward(retain_graph=True)

        if args.mode == "kd":
            scaler.scale(loss + loss_s).backward()
        else:
            scaler.scale(loss_s).backward()

        if args.mode == "kd":
            total_loss += loss.item() + loss_s.item()
            cur_loss += loss.item() + loss_s.item()
        else: 
            total_loss += loss_s.item()
            cur_loss += loss_s.item()
        
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        if idx%args.log_interval==(args.log_interval-1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60))
            cur_loss = 0.0
            sys.stdout.flush()
    
    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)

def save_state_dict(student , teacher, args):
    if torch.cuda.device_count() > 1: 
        if args.mode == "kd":
            params = {
                'student': student.module.state_dict(),
                'teacher': teacher.module.state_dict()
            }
        else :
            params = {
                'student': student.module.state_dict()
            }
    else :
        if args.mode == "kd":
            params = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict()
            }
        else :
            params = {
                'student': student.state_dict(),
            }
    torch.save(params, args.model_val_path)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(0, args.no_epochs):
    loss = train(student, optimizer , train_loader, epoch, device, args, teacher)

    if args.lr_sched:
        scheduler.step()

    with torch.no_grad():
        if args.mode == "kd":
            _ = validate(teacher, val_loader, epoch, device)
            cc_loss = validate(student, val_loader, epoch, device)
        else :
            cc_loss = validate(student, val_loader, epoch, device)

        if epoch == 0 :
            best_loss = cc_loss
        if best_loss <= cc_loss:
            best_loss = cc_loss
            print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
            save_state_dict(student , teacher, args)
        print()