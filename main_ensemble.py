from loss import *
import data
from argparse import ArgumentParser
from models.backbone import *
import torch
import torch.nn as nn
from utils import AverageMeter
import time
import sys
from ptflops import get_model_complexity_info
import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset_dir", type=str, default="data/")
parser.add_argument('--input_size_h',default=256, type=int)
parser.add_argument('--input_size_w',default=256, type=int)
parser.add_argument('--no_workers',default=8, type=int)
parser.add_argument('--no_epochs',default=10, type=int)
parser.add_argument('--log_interval',default=20, type=int)
parser.add_argument('--lr_sched',default=True, type=bool)
parser.add_argument('--model_val_path',default="model.pt", type=str)
parser.add_argument('--model_salicon_path',default="model_salicon.pt", type=str)

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

parser.add_argument('--student',default="eeeac2", type=str)
parser.add_argument('--teacher',default="ofa595", type=str)
parser.add_argument('--teacher2',default="efb7", type=str)

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

if args.teacher2 == "ofa595":
    teacher2 = OFA595(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher2 == "tresnet":
    teacher2 = tresnet(num_channels=3, train_enc=True, load_weight=1, pretrained='1k', output_size=args.output_size)
elif args.teacher2 == "mbv3":
    teacher2 = MobileNetV3_1k(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher2 == "efb0":
    teacher2 = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher2 == "efb4":
    teacher2 = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher2 == "efb7":
    teacher2 = EfficientNetB7(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher2 == "pnas":
    teacher2 = PNASModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
elif args.teacher == "vgg":
    teacher2 = VGGModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)

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

if args.dataset != "salicon":
    model_load_state_dict(student , teacher, args.model_salicon_path)

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
        teacher2 = nn.DataParallel(teacher2)

student.to(device)

if args.mode == "kd":
    teacher.to(device)
    teacher2.to(device)
else: 
    teacher = None

if args.mode == "kd":
    params_group = [
        {"params": list(filter(lambda p: p.requires_grad, teacher.parameters())), "lr" : args.learning_rate },
        {"params": list(filter(lambda p: p.requires_grad, teacher2.parameters())), "lr" : args.learning_rate },
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
    
    return cc_loss.avg + (nss_loss.avg/10) + sim_loss.avg + (1-kldiv_loss.avg) + auc_loss.avg

def train(student, optimizer, loader, epoch, device, args, teacher, teacher2):
    student.train()
    if args.mode == "kd":
        teacher.train()
        teacher2.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()

        if args.mode == "kd":
            if args.mixed:
                with torch.cuda.amp.autocast():
                    pred_map = teacher(img)
                    loss = loss_func(pred_map, gt, fixations, args)

                    pred_map2 = teacher2(img)
                    loss2 = loss_func(pred_map2, gt, fixations, args)

                    mean_pred = (pred_map+pred_map2)/2
            else :
                pred_map = teacher(img)
                loss = loss_func(pred_map, gt, fixations, args)

        with torch.cuda.amp.autocast():
            pred_map_student = student(img)
            if args.mode == "kd":
                loss_s = loss_func(pred_map_student, mean_pred.detach(), fixations, args)
            else :
                loss_s = loss_func(pred_map_student, gt, fixations, args)
        
        if args.mode == "kd":
            wts = 0.5
            # eq.6
            #combined_loss = (wts * loss) + ((1-wts) * loss_s)
            combined_loss = loss + loss2 + loss_s
            if args.mixed:
                scaler.scale(combined_loss).backward()
            else :
                combined_loss.backward()
        else:
            if args.mixed:
                scaler.scale(loss_s).backward()
            else: 
                loss_s.backward()

        if args.mode == "kd":
            total_loss += combined_loss.item()
            cur_loss += combined_loss.item()
        else: 
            total_loss += loss_s.item()
            cur_loss += loss_s.item()

        if args.mixed:
            scaler.step(optimizer)
            scaler.update()
        else :
            optimizer.step()

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

start= time.time()

for epoch in range(0, args.no_epochs):
    loss = train(student, optimizer , train_loader, epoch, device, args, teacher, teacher2)

    if args.lr_sched:
        scheduler.step()

    with torch.no_grad():
        if args.mode == "kd":
            _ = validate(teacher, val_loader, epoch, device)
            _ = validate(teacher2, val_loader, epoch, device)
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

end = time.time()
print("Time = ",  ((end - start))/60, " Min")
