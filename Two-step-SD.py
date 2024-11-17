import warnings
import torch, math, time, argparse, os
import random, dataset, utils, losses, net
import numpy as np
from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from net.alexnet import *
from net.densenet import *
from net.effitionnet import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate
from tqdm import *
from torchvision.models import efficientnet
import torchvision.transforms as transforms
from dataset.cifair import *
import torchvision.datasets as datasets

# os.environ['CUDA_VISIBLE_DEVICES']='1'
#参数设置

args_imb_type = 'exp'
args_imb_factor = 0.02  #不平很率
args_rand_number = 0
args_workers = 1
args_emb_size=512
# args_sz_batch=90 #批处理得数量
args_sz_batch = 300 #wudi
# args_sz_batch = 30
args_nb_workers=0
args_gpu_id=1
# args_lr=1e-7  #学习率
args_lr = 1e-4
args_nb_epochs=100  #迭代次数
args_bn_freeze=1 #是否冻结
# args_gpu_id=0
args_warm=1  #是否热训练
# args_LOG_DIR='../logs'
# args_dataset='kongyu'
# args_dataset = 'cub'
args_dataset = 'cifar10'
# args_model='bn_inception'
# args_model = 'resnet50'
# args_model = 'googlenet'
# args_model = 'resnet101' 
# args_model = 'alexnet'
# args_model = 'effitionnet'
args_model = 'densenet121'
# args_model = 'densenet161'
# args_loss='Contrastive'
# args_loss = 'MS'
args_loss = 'Triplet'
# args_loss = 'Proxy_NCA'
# args_loss='New_loss'
# args_loss = 'Circle'
# args_loss = 'Proxy_Anchor'
# args_loss = 'NPair'
args_sz_embedding=512
args_alpha=32
args_mrg=0.4
args_optimizer='adamw'
args_remark=''
args_IPC=''
args_l2_norm=1
# args_bn_freeze=1
args_weight_decay=1e-4
args_lr_decay_step=10
# args_lr_decay_step = 3
args_lr_decay_gamma=0.5
# args_evald='confusion'
args_evald = 'acc'



#设置随机种子便于复现
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus



torch.cuda.set_device(1)

# Directory for Log
# LOG_DIR = args_LOG_DIR + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args_dataset, args_model, args_loss, args_sz_embedding, args_alpha, 
#                                                                                             args_mrg, args_optimizer, args_lr, args_sz_batch, args_remark)

# os.chdir('../data/')
data_root = os.getcwd()
# Dataset Loader and Sampler
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomCrop(40, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    
if args_dataset == 'cifar10':
    train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args_imb_type, imb_factor=args_imb_factor, rand_number=args_rand_number, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
elif args_dataset == 'cifar100':
    train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args_imb_type, imb_factor=args_imb_factor, rand_number=args_rand_number, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
else:
    warnings.warn('Dataset is not listed')

cls_num_list = train_dataset.get_cls_num_list()
print('cls num list:')
print(cls_num_list)
print('test_cls:')
print(val_dataset.test_list)
args_cls_num_list = cls_num_list

train_sampler = None
    
dl_tr = torch.utils.data.DataLoader(
    train_dataset, batch_size=args_sz_batch, shuffle=(train_sampler is None),
    num_workers=args_workers, pin_memory=True, sampler=train_sampler)

dl_ev = torch.utils.data.DataLoader(
    val_dataset, batch_size=100, shuffle=False,
    num_workers=args_workers, pin_memory=True)

nb_classes = 10

# Backbone Model
if args_model.find('googlenet')+1:
    model = googlenet(embedding_size=args_sz_embedding, pretrained=True, is_norm=args_l2_norm, bn_freeze = args_bn_freeze)
elif args_model.find('bn_inception')+1:
    model = bn_inception(embedding_size=args_sz_embedding, pretrained=True, is_norm=args_l2_norm, bn_freeze = args_bn_freeze)
elif args_model.find('resnet18')+1:
    model = Resnet18(embedding_size=args_sz_embedding, pretrained=True, is_norm=args_l2_norm, bn_freeze = args_bn_freeze)
elif args_model.find('resnet50')+1:
    model = Resnet50(embedding_size=args_sz_embedding, pretrained=False, is_norm=args_l2_norm, bn_freeze = args_bn_freeze)
elif args_model.find('resnet101')+1:
    model = Resnet101(embedding_size=args_sz_embedding, pretrained=True, is_norm=args_l2_norm, bn_freeze = args_bn_freeze)
elif args_model.find('alexnet')+1:
    model = alexnet(embedding_size=args_sz_embedding, pretrained=True, is_norm=args_l2_norm, bn_freeze = args_bn_freeze)
elif args_model.find('densenet121')+1:
    model = Densenet121(embedding_size=args_sz_embedding, pretrained=True, is_norm=args_l2_norm, bn_freeze = args_bn_freeze)
elif args_model.find('densenet161')+1:
    model = Densenet161(embedding_size=args_sz_embedding, pretrained=True, is_norm=args_l2_norm, bn_freeze = args_bn_freeze)
elif args_model.find('effitionnet')+1:
    model = efficientnet.efficientnet_v2_s(True)
    model.classifier = nn.Linear(1280,5)
model = model.cuda()

if args_gpu_id == -1:
    model = nn.DataParallel(model)

# DML Losses
if args_loss == 'Proxy_Anchor':
    criterion = losses.Proxy_Anchor(nb_classes = nb_classes, sz_embed = args_sz_embedding, mrg = args_mrg, alpha = args_alpha).cuda()
elif args_loss == 'Proxy_NCA':
    criterion = losses.Proxy_NCA(nb_classes = nb_classes, sz_embed = args_sz_embedding).cuda()
elif args_loss == 'MS':
    criterion = losses.MultiSimilarityLoss().cuda()
elif args_loss == 'Contrastive':
    criterion = losses.ContrastiveLoss(mrg = args_mrg).cuda()
elif args_loss == 'Triplet':
    criterion = losses.TripletLoss(mrg = args_mrg).cuda()
elif args_loss == 'NPair':
    criterion = losses.NPairLoss().cuda()
elif args_loss == 'Rank_List_Proxy_Anchor':
    criterion = losses.Rank_List_Proxy_Anchor(nb_classes = nb_classes, sz_embed = args_sz_embedding).cuda()
elif args_loss == 'Rank_List_Proxy_Anchor2':
    criterion = losses.Rank_List_Proxy_Anchor2(nb_classes = nb_classes, sz_embed = args_sz_embedding).cuda()
elif args_loss == 'DA_Rank_List_Proxy_Anchor':
    criterion = losses.DA_Rank_List_Proxy_Anchor(nb_classes = nb_classes, sz_embed = args_sz_embedding).cuda()
elif args_loss == 'EE_Rank_List_Proxy_Anchor':
    criterion = losses.EE_Rank_List_Proxy_Anchor(nb_classes = nb_classes, sz_embed = args_sz_embedding).cuda()
elif args_loss == 'New_loss':
    criterion = losses.New_loss(nb_classes = nb_classes, sz_embed = args_sz_embedding).cuda()
elif args_loss == 'Circle':
    criterion = losses.Circle().cuda()
    

# Train Parameters
param_groups = [
    {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args_gpu_id != -1 else 
                 list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
    {'params': model.model.embedding.parameters() if args_gpu_id != -1 else model.module.model.embedding.parameters(), 'lr':float(args_lr) * 1},
]
if args_loss == 'Proxy_Anchor':
    param_groups.append({'params': criterion.proxies, 'lr':float(args_lr) * 100})
elif args_loss == 'Rank_List_Proxy_Anchor':
    param_groups.append({'params': criterion.proxies, 'lr': float(args_lr) * 100})
elif args_loss == 'Rank_List_Proxy_Anchor2':
    param_groups.append({'params': criterion.proxies, 'lr': float(args_lr) * 100})
elif args_loss == 'New_loss':
    param_groups.append({'params': criterion.proxies, 'lr': float(args_lr) * 100})
elif args_loss == 'DA_Rank_List_Proxy_Anchor':
    param_groups.append({'params': criterion.proxies, 'lr': float(args_lr) * 100})
    param_groups.append({'params': criterion.alphac})
elif args_loss == 'EE_Rank_List_Proxy_Anchor':
    param_groups.append({'params': criterion.proxies, 'lr': float(args_lr) * 100})

# Optimizer Setting
if args_optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=float(args_lr), weight_decay = args_weight_decay, momentum = 0.9, nesterov=True)
elif args_optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(args_lr), weight_decay = args_weight_decay)
elif args_optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args_lr), alpha=0.9, weight_decay = args_weight_decay, momentum = 0.9)
elif args_optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args_lr), weight_decay = args_weight_decay)

# opt = torch.optim.Adam(model.parameters(), lr=float(args_lr), weight_decay = args_weight_decay)

#包装model和opt
# model, opt = amp.initialize(model, opt, opt_level="O1")
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args_lr_decay_step, gamma = args_lr_decay_gamma)

# print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args_nb_epochs))
losses_list = []
# acc_list = []
acc_list = '\n' +'Acc_list: '
best_recall=[0]
best_acc = 0
best_mae = 1
best_f1 = 0
best_epoch = 0
start = time.time()
print("start training")

log_testing = open('log/SD_Two_step_{model}_{loss}_{type}_{rho}_7_3.csv'.format(model=args_model,loss=args_loss,type=args_imb_type,rho=str(args_imb_factor)), 'a')
log_testing.write('\n'+'----------------------------------------------------'+"\n"+'{data}'.format(data=time.ctime())+"\n"+'----------------------------------------------------'+'\n')
log_testing.flush()

# log_acc = ''
# log_C_1 = ''
# log_C_2 = ''
# log_C_3 = ''
# log_C_4 = ''
# log_C_5 = ''
max_acc=0


for epoch in range(0, args_nb_epochs):
    model.train()
    bn_freeze = args_bn_freeze
    if bn_freeze:
        modules = model.model.modules() if args_gpu_id != -1 else model.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []
    acc = 0.
    mae = 0.
    f1 = 0.
    
    
    # Warmup: Train only new params, helps stabilize learning.
    if args_warm > 0:
        if args_gpu_id != -1:
            unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
        else:
            unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())

        if epoch == 0:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == args_warm:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:
        m = model(x.squeeze().cuda())
        if args_loss == 'DA_Rank_List_Proxy_Anchor':
            feature = model.feature
            loss = criterion(m, y.squeeze().cuda(),feature)
        else:
            loss = criterion(m, y.squeeze().cuda())

        opt.zero_grad()
        loss.backward()

        #混合精度
        # with amp.scale_loss(loss, opt) as scaled_loss:
        #     scaled_loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        if args_loss == 'Proxy_Anchor':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        elif args_loss == 'Rank_List_Proxy_Anchor2':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        elif args_loss == 'DA_Rank_List_Proxy_Anchor':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        elif args_loss == 'EE_Rank_List_Proxy_Anchor':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        elif args_loss == 'New_loss':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

        
    losses_list.append(np.mean(losses_per_epoch))
    scheduler.step()
    
    if(epoch >= 0):
        with torch.no_grad():
            # print("**Evaluating...**")
            if args_evald == 'recallk':
                Recalls = utils.evaluate_cos(model, dl_ev)
            elif args_evald == 'acc':
                acc = utils.evaluate_acc(model, dl_ev)
                
                if acc>max_acc:
                    max_acc = acc
                print("New_epoch:{e},Acc={acc}".format(e=epoch+1,acc=acc))
                print('Max_acc: '+str(round(max_acc,4)))
                log_testing.write('epoch: '+str(epoch+1) +' '+ 'Acc: '+ str(round(acc,4)) + ' ')
                log_testing.write('Max_acc: '+str(round(max_acc,4))+ '\n')
                log_testing.flush()
                acc_list+= str(acc)+' '
                # acc_list.append(acc)
            elif args_evald == 'confusion':
                confusion,acc = utils.evaluate_confusion(model,dl_ev)
                print('confusion_matrix is\n',confusion)
                # log_testing.write('epoche: '+ str(epoch+1) +' '+ 'Acc: ' +str(round(acc,2)) +'\n'+str(confusion)+'\n')
                # log_testing.flush()
                # log_acc+=str(round(acc,4))+' '
                # log_C_1+=str(confusion[0][0])+' '
                # log_C_2+=str(confusion[1][1])+' '
                # log_C_3+= str(confusion[2][2])+' '
                # log_C_4+= str(confusion[3][3])+' '
                # log_C_5+= str(confusion[4][4])+' '

            elif args_evald == 'mae':
                mae = utils.evaluate_mae(model, dl_ev)
            else:
                f1 = utils.evaluate_small(model, dl_ev)

log_testing.write(acc_list)
log_testing.flush()
log_testing.close()
# log_testing.write('Acc: '+log_acc+'\n')
# log_testing.write('C1: '+log_C_1+'\n')
# log_testing.write('C2: '+log_C_2+'\n')
# log_testing.write('C3: '+log_C_3+'\n')
# log_testing.write('C4: '+log_C_4+'\n')
# log_testing.write('C5: '+log_C_5+'\n')
# log_testing.flush()
# log_testing.close()