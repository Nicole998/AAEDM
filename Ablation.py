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
import time



#参数设置
args_emb_size=512
args_sz_batch = 90
args_nb_workers=0
args_gpu_id=1
args_lr=1e-7  #学习率

args_nb_epochs=10
args_bn_freeze=1 #是否冻结

args_warm=1  #是否热训练
args_LOG_DIR='../logs'
args_dataset='kongyu'
# args_dataset = 'cub'
args_model='bn_inception'
# args_model = 'resnet18'
# args_model = 'resnet50'
# args_model = 'resnet32'
# args_model = 'googlenet'
# args_model = 'resnet101'
# args_model = 'alexnet'
# args_model = 'effitionnet'
# args_model = 'densenet121'
# args_model = 'densenet161'
args_loss='New_loss'



args_sz_embedding=512
args_alpha=32
args_mrg=0.4
args_optimizer='adamw'
args_remark=''
args_IPC=''
args_l2_norm=1
# args_bn_freeze=1
args_weight_decay=1e-4
args_lr_decay_step=3
# args_lr_decay_step = 3
args_lr_decay_gamma=0.5
# args_evald='mae'
# args_evald = 'acc'
args_evald='all'

print('\n')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print('Model: ',args_model)
#设置随机种子便于复现

# seed = 1
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) # set random seed for all gpus


## 设置使用GPU
torch.cuda.set_device(1)
data_root = os.getcwd()

###训练集
trn_dataset = dataset.load(
    name=args_dataset,
    root=data_root,
    mode='train',
    transform=dataset.utils.make_transform(
        is_train=True,
        is_inception=(args_model == 'bn_inception')
    )
)

dl_tr = torch.utils.data.DataLoader(
    trn_dataset,
    batch_size=args_sz_batch,
    shuffle=True,
    num_workers=args_nb_workers,
    drop_last=True,
    pin_memory=True
)

### 测试集
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



##设置模型
model = alexnet(embedding_size=args_sz_embedding, pretrained=True, is_norm=args_l2_norm, bn_freeze = args_bn_freeze)
model = model.cuda()


# 研究损失消融
nb_classes = 5
# criterion = losses.New_loss(nb_classes = nb_classes, sz_embed = args_sz_embedding,lamda = 0.1).cuda()
criterion = losses.New_loss1().cuda()


# Train Parameters
param_groups = [
    {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args_gpu_id != -1 else 
                 list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
    {'params': model.model.embedding.parameters() if args_gpu_id != -1 else model.module.model.embedding.parameters(), 'lr':float(args_lr) * 1},
]

# 设置参数
# param_groups.append({'params': criterion.proxies, 'lr': float(args_lr) * 100})


# Optimizer Setting 优化器
opt = torch.optim.Adam(param_groups, lr=float(args_lr), weight_decay = args_weight_decay)


#包装model和opt
### 设置学习率衰减
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

log_acc = ''
log_C_1 = ''
log_C_2 = ''
log_C_3 = ''
log_C_4 = ''
log_C_5 = ''



## 训练模型
for epoch in range(0, args_nb_epochs):
    model.train()
    bn_freeze = args_bn_freeze
    if bn_freeze:
        modules = model.model.modules() if args_gpu_id != -1 else model.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    
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
        # loss = criterion(m, y.squeeze().cuda(),epoch=epoch)
        loss = criterion(m, y.squeeze().cuda())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 15)##梯度裁剪
        torch.nn.utils.clip_grad_value_(criterion.parameters(), 15)
        # losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()
        # losses_list.append(np.mean(losses_per_epoch))
        scheduler.step()
    

    ## 测试部
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
