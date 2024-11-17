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
# from torchvision.models import efficientnet
import time



# parser = argparse.ArgumentParser(description=
#     'Loss is implement by LINGHUI'
# )

# parser.add_argument('--loss', 
#     default = 'New_loss', 
#     help = 'Loss function.'
# )

# args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES']='1'
#参数设置
args_emb_size=512
# args_sz_batch=90 #批处理得数量
args_sz_batch = 90
# args_sz_batch = 30
args_nb_workers=0
args_gpu_id=1
args_lr=1e-4  #学习率
# args_lr = 1e-4
args_nb_epochs=10
args_bn_freeze=1 #是否冻结
# args_gpu_id=0
args_warm=1  #是否热训练
args_LOG_DIR='/root/autodl-tmp/trafficPoject/logs'
args_dataset='kongyu'
# args_dataset = 'cub'
# args_model='bn_inception'
# args_model = 'resnet18'
# args_model = 'resnet50'
args_model = 'resnet32'
# args_model = 'googlenet'
# args_model = 'resnet101'
# args_model = 'alexnet'
# args_model = 'effitionnet'
# args_model = 'densenet121'
# args_model = 'densenet161'
# args_loss='Contrastive'
# args_loss = 'MS'
# args_loss = 'Triplet'
# args_loss = 'Proxy_NCA'
# args_loss='New_loss'
args_loss = 'Circle'
# args_loss = 'Proxy_Anchor'
# args_loss = 'NPair'
# args_loss = 'Rank_List_Proxy_Anchor'
# args_loss = 'Rank_List_Proxy_Anchor2'


args_sz_embedding=512
args_alpha=32
args_mrg=0.4
args_optimizer='adamw'
args_remark=''
args_IPC=''
args_l2_norm=1
# args_bn_freeze=1
args_weight_decay=1e-4
args_lr_decay_step=5
# args_lr_decay_step = 3
args_lr_decay_gamma=0.5
# args_evald='mae'
# args_evald = 'acc'
args_evald='all'

print('\n')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

#设置随机种子便于复现
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus



torch.cuda.set_device(0)


# os.chdir('../data/')
data_root = os.getcwd()
# Dataset Loader and Sampler
if args_dataset != 'Inshop':
    trn_dataset = dataset.load(
            name = args_dataset,
            root = data_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True, 
                is_inception = (args_model == 'bn_inception')
            ))
else:
    trn_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True, 
                is_inception = (args_model == 'bn_inception')
            ))

if args_IPC:
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args_sz_batch, images_per_class = args_IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = args_sz_batch, drop_last = True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args_nb_workers,
        pin_memory = True,
        batch_sampler = batch_sampler
    )
    print('Balanced Sampling')
    
else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size = args_sz_batch,
        shuffle = True,
        num_workers = args_nb_workers,
        drop_last = True,
        pin_memory = True
    )
    # print('Random Sampling')

if args_dataset != 'Inshop':
    ev_dataset = dataset.load(
            name = args_dataset,
            root = data_root,
            mode = 'eval',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args_model == 'bn_inception')
            ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size = args_sz_batch,
        shuffle = False,
        num_workers = args_nb_workers,
        pin_memory = True
    )
    
else:
    query_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'query',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args_model == 'bn_inception')
    ))
    
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size = args_sz_batch,
        shuffle = False,
        num_workers = args_nb_workers,
        pin_memory = True
    )

    gallery_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'gallery',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args_model == 'bn_inception')
    ))
    
    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size = args_sz_batch,
        shuffle = False,
        num_workers = args_nb_workers,
        pin_memory = True
    )

nb_classes = 5

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
    criterion = losses.New_loss(nb_classes = nb_classes, sz_embed = args_sz_embedding,lamda = 0.1).cuda()
elif args_loss == 'Circle':
    criterion = losses.Circle().cuda()

print('loss :',args_loss)

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
acc_list = []
best_recall=[0]
best_acc = 0
best_mae = 1
best_f1 = 0
best_epoch = 0
start = time.time()
print("start training")

# log_testing = open('log/Two_step_{model}_{loss}_7_3.csv'.format(model=args_model,loss=args_loss), 'a')
# log_testing.write('----------------------------------------------------'+"\n"+'{data}'.format(data=time.ctime())+"\n"+'----------------------------------------------------'+'\n')
# log_testing.flush()

log_acc = ''
log_C_1 = ''
log_C_2 = ''
log_C_3 = ''
log_C_4 = ''
log_C_5 = ''


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
        if args_loss != 'New_loss':
            # feature = model.feature
            loss = criterion(m, y.squeeze().cuda())
        else:
            loss = criterion(m, y.squeeze().cuda(),epoch=epoch)

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
                if best_acc<acc:
                    best_acc = acc
                print("New_epoch:{e},Acc={acc}".format(e=epoch+1,acc=best_acc))
                acc_list.append(acc)
            elif args_evald == 'confusion':
                confusion,acc = utils.evaluate_confusion(model,dl_ev)
                print('epoch=>',epoch)
                print('confusion_matrix is\n',confusion)
                # log_testing.write('epoche: '+ str(epoch+1) +' '+ 'Acc: ' +str(round(acc,4)) +'\n'+str(confusion)+'\n')
                # log_testing.flush()
                log_acc+=str(round(acc,4))+' '
                log_C_1+=str(confusion[0][0])+' '
                log_C_2+=str(confusion[1][1])+' '
                log_C_3+= str(confusion[2][2])+' '
                log_C_4+= str(confusion[3][3])+' '
                log_C_5+= str(confusion[4][4])+' '

            elif args_evald == 'mae':
                mae = utils.evaluate_mae(model, dl_ev)
            elif args_evald == 'all':
                all_metric = utils.evaluate_acc_mae_f1(model, dl_ev)
            else:
                f1 = utils.evaluate_small(model, dl_ev)

# log_testing.write('Acc: '+log_acc+'\n')
# log_testing.write('C1: '+log_C_1+'\n')
# log_testing.write('C2: '+log_C_2+'\n')
# log_testing.write('C3: '+log_C_3+'\n')
# log_testing.write('C4: '+log_C_4+'\n')
# log_testing.write('C5: '+log_C_5+'\n')
# log_testing.flush()
# log_testing.close()