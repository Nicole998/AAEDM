import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from collections import defaultdict
from pytorch_metric_learning import miners, losses
from hyptorch.pmath import dist_matrix

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def get_mean(X, T):
    #X_numpy = X.detach().cpu().numpy()
    label = T
    X_mean = torch.zeros_like(X)
    index = get_cindex(T)
    len1 = len(index)
    for i in range(len1):
        len2 = len(index[i])
        for j in index[i]:
            X_mean[i] += X[j]
        X_mean[i] / len2
    # for i in range(len(label)):
    #     X_mean[i][:]=X_numpy[i][:]
    #     c_nums = 1
    #     for j in range(len(label)):
    #         if i == j:
    #             continue
    #         else:
    #             if label[i] == label[j]:
    #                 X_mean[i][:]=X_mean[i][:] + X_numpy[j][:]
    #                 c_nums += 1
    #     X_mean[i][:] / c_nums
        #cn.extend([[c_nums]*(np.shape(X_numpy)[1])])
    return X_mean

def get_cindex(T):
    # t = list(set(T2N))
    # t.sort(key = T2N.index)
    # index = []
    # for i in range(len(t)):
    #     idx = []
    #     for j in range(len(T2N)):
    #         if t[i] == T2N[j]:
    #             idx.append(j)
    #     index.append(idx)
    d = defaultdict(list)
    for i, v in enumerate(T):
        d[v].append(i)
    return list(d.values())

def get_label(Y):
    y = list(set(Y))
    y.sort(key = Y.tolist().index)
    return y

def get_y_ac(y,ac):
    ac2np = ac.detach().cpu().numpy()
    y_ac = np.zeros_like(y)
    for i in range(len(y)):
        y_ac[i] = ac2np[y[i]]
    return torch.FloatTensor(y_ac).cuda()

  
def get_D_avg(X,m,index):
    davg = []
    len1 = len(index)
    # for i in range(X.shape[0]):
    #     num = torch.norm(X[i]-m[i],2)
    #     D_avg.append(math.pow(num,2))
    # for i in range(len1):
    #     n = 0
    #     len2 = len(index[i])
    #     for j in range(len2):
    #         n += D_avg[index[i][j]]
    #     davg.append(n / len2)
    for i in range(len1):
        num = 0
        len2 = len(index[i])
        for j in index[i]:
            num += F.pairwise_distance(X[j,:].unsqueeze(0), m[i,:].unsqueeze(0))
        davg.append(num / len2)
    return torch.FloatTensor(davg).cuda()

def getDA(davg,ac,y):
    num1 = 0
    num2 = 0
    for i in range(len(y)):
        num1 += math.pow((davg[i] - ac[y[i]]),2)
        num2 += ac[y[i]]
    return num1,num2
  
def Inter_Class_Constraint(davg0, ac, y, eta=0.5):
    davg0_eta = torch.pow(davg0,eta)
    c = len(y)
    summ = 0
    for i in range(c):
        for j in range(c):
            davg0_ac_j_i = davg0_eta[j] * ac[y[i]]
            davg0_ac_i_j = davg0_eta[i] * ac[y[j]]
            summ += math.pow((davg0_ac_j_i - davg0_ac_i_j),2)
    summ = summ / 2
    return summ / math.pow(c,2)
  
class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss


class Rank_List_Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, m=1.4):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.m = m
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        # cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        # cos_dis = 1-cos

        cos_dis = dist_matrix(l2_norm(X),l2_norm(P),c=2)
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        #负对权重
        t = 10
        W = torch.exp(t*(self.m - cos_dis))
        W_sum = torch.where(N_one_hot == 1, W, torch.zeros_like(W)).sum(dim=0)
        W_sum = W_sum.sum()

        pos_exp = torch.exp(self.alpha * (cos_dis - (self.m - self.mrg)))
        neg_exp = (W/W_sum)*torch.exp(self.alpha * (self.m - cos_dis))

        # pos_exp = torch.clamp((cos_dis - (self.m - self.mrg)),min=0)
        # neg_exp = (W/W_sum)*torch.clamp((self.m - cos_dis),min=0)

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

        # pos_term = P_sim_sum.sum() / num_valid_proxies
        # neg_term = N_sim_sum.sum() / self.nb_classes

        loss = pos_term + neg_term

        return loss

class Rank_List_Proxy_Anchor2(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.8, alpha=32, m=1.4):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.m = m
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        cos_dis = 1 - cos
        #new add part
        #++++++++++++++++++++++++++++++++++
        cos_dis=abs(cos_dis-0.2)
        #++++++++++++++++++++++++++++++++
        #print("####l2_norm(X):",l2_norm(X).shape)
        #print("####l2_norm(P):",l2_norm(P).shape)
        #print("####cos:",cos.shape)
        # cos_dis = F.pairwise_distance(l2_norm(X),l2_norm(P),p=2)
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        #print("####P:",P.shape)
        #print("####X:",X.shape)
        #print("####T:",T.shape)
        #print("####nb_classes:",self.nb_classes)
        # print("p_onhot:  ",P_one_hot)
        # print("n_onhot:  ", N_one_hot)
        # print("cos_dis: ",cos_dis)
        # 负对权重
        
        t = 20
        W = torch.exp(t * (self.m - cos_dis))
        W_neg = torch.where(N_one_hot == 1, W, torch.zeros_like(W))
        W_sum0 = W_neg.sum(dim=0)

        # pos_exp = torch.exp(self.alpha * (cos_dis - (self.m - self.mrg)))
        # neg_exp = (W/W_sum)*torch.exp(self.alpha * (self.m - cos_dis))

        pos_exp = torch.clamp((cos_dis - (self.m - self.mrg)), min=0)
        neg_exp = (W_neg/W_sum0)*torch.clamp((self.m - cos_dis), min=0)

        # print("pos_exp:  ",pos_exp)
        # print("neg_exp:  ", neg_exp)
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
                dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        # pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        # neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

        pos_term = P_sim_sum.sum() / num_valid_proxies
        neg_term = N_sim_sum.sum() / self.nb_classes

        loss = pos_term + neg_term

        return loss

class DA_Rank_List_Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.4, m=1.4, lamda=10):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        # alphac initialization
        #self.alphac =  torch.nn.Parameter(torch.nn.init.constant_(torch.randn(nb_classes).cuda(),0.5))
        a_init = torch.full((nb_classes,),0.5).cuda()
        self.alphac =  torch.nn.Parameter(a_init,requires_grad=True)

        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.m = m
        self.mrg = mrg
        self.lamda = lamda

    def forward(self, X, T, Feature):
        P = self.proxies
        
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        cos_dis = 1 - cos
        # cos_dis = F.pairwise_distance(l2_norm(X),l2_norm(P),p=2)
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        #print("####P:",l2_norm(X).shape)
        # print("####P:",P.shape)
        # print("####X:",X.shape)
        #print("####T:",T,T.shape)
        # print("####nb_classes:",self.nb_classes)
        
        #print("p_onhot:  ",P_one_hot.shape)
        #a1,p = torch.where(P_one_hot)
        #a2,n = torch.where(N_one_hot)
        #print("a1",a1,a1.shape)
        #print("p",p,p.shape)
        #print("a2",a2,a2.shape)
        #print("n",n,n.shape)
        #print("n_onhot:  ", N_one_hot.shape)
        #print("cos_dis: ",cos_dis)
        # 负对权重
        
        t = 20
        W = torch.exp(t * (self.m - cos_dis))
        W_neg = torch.where(N_one_hot == 1, W, torch.zeros_like(W))
        W_sum0 = W_neg.sum(dim=0)

        # pos_exp = torch.exp(self.alpha * (cos_dis - (self.m - self.mrg)))
        # neg_exp = (W/W_sum)*torch.exp(self.alpha * (self.m - cos_dis))

        pos_exp = torch.clamp((cos_dis - (self.m - self.mrg)), min=0)
        neg_exp = (W_neg/W_sum0)*torch.clamp((self.m - cos_dis), min=0)

        # print("pos_exp:  ",pos_exp)
        # print("neg_exp:  ", neg_exp)
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
                dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        # pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        # neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

        pos_term = P_sim_sum.sum() / num_valid_proxies
        neg_term = N_sim_sum.sum() / self.nb_classes

        # DA-DML
        T2N = T.cpu().numpy()
        T2N.sort()
        L2N_X = l2_norm(X)
        ac = self.alphac
        batch_y = get_label(T2N)
        #print("batch_y:",batch_y)
        #umean=get_mean(X, T2N)
        #umean = l2_norm(self.proxies)
        cindex = get_cindex(T2N)
        #print("cindex:",cindex,len(cindex))
        D_avg = get_D_avg(L2N_X,l2_norm(P),cindex)
        
        #num1 = 0
        #num2 = 0
        #for i in range(len(batch_y)):
        #   num1 += math.pow((D_avg[i] - ac[batch_y[i]]),2)
        #    num2 += ac[batch_y[i]]
        num1,num2 = getDA(D_avg,ac,batch_y)
        D_ac_sum = num1 / self.nb_classes
        
        # Inter-class Density Correlations Preservation Constraint
        L2N_F = l2_norm(Feature)
        umean2 =get_mean(Feature,T2N)
        davg0 = get_D_avg(L2N_F,l2_norm(umean2),cindex)
        inter_class = Inter_Class_Constraint(davg0,ac,batch_y)
        
        LDA = D_ac_sum - (num2 / self.nb_classes) + inter_class
        #print("####D_avg:",D_avg,D_avg.shape)
        #print("####sc:",sc,sc.shape)

        loss = pos_term + neg_term + self.lamda * LDA

        return loss

      
def get_embedding_aug(embeddings, labels, num_instance, n_inner_pts, isl2_norm=True):
    batch_size = embeddings.shape[0]
    
    assert num_instance % 2 == 0, 'num_instance should be even number for simple implementation'
    swap_axes_list = [i + 1 if i % 2 == 0 else i - 1 for i in range(batch_size)]
    swap_embeddings = embeddings[swap_axes_list]
    pos = embeddings
    anchor = swap_embeddings
    concat_embeddings = embeddings.clone()
    concat_labels = labels.clone()
    n_pts = n_inner_pts
    l2_normalize = isl2_norm
    total_length = float(n_pts + 1)
    for n_idx in range(n_pts):
        left_length = float(n_idx + 1)
        right_length = total_length - left_length
        inner_pts = (anchor * left_length + pos * right_length) / total_length
        if l2_normalize:
            inner_pts = l2_norm(inner_pts)
        concat_embeddings = torch.cat((concat_embeddings, inner_pts), dim=0)
        concat_labels = torch.cat((concat_labels, labels), dim=0)

    return concat_embeddings, concat_labels


class EE_Rank_List_Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.4, m=1.4, num_instances=2, n_inner_pts=1, isl2_norm=True):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(
            torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.m = m
        self.mrg = mrg
        self.num_instance = num_instances
        self.n_inner_pts = n_inner_pts
        self.isl2_norm = isl2_norm

    def forward(self, X, T):
        P=self.proxies

        if self.n_inner_pts != 0:
            embeddings, labels = get_embedding_aug(X, T, self.num_instance, self.n_inner_pts, self.isl2_norm)

        cos=F.linear(l2_norm(embeddings), l2_norm(P))  # Calcluate cosine similarity
        cos_dis=1 - cos
        # cos_dis = F.pairwise_distance(l2_norm(X),l2_norm(P),p=2)
        P_one_hot=binarize(T=labels, nb_classes=self.nb_classes)
        N_one_hot=1 - P_one_hot

        

        t=20
        W=torch.exp(t * (self.m - cos_dis))
        W_neg=torch.where(N_one_hot == 1, W, torch.zeros_like(W))
        W_sum0=W_neg.sum(dim=0)

        pos_exp=torch.clamp((cos_dis - (self.m - self.mrg)), min=0)
        neg_exp=(W_neg/W_sum0)*torch.clamp((self.m - cos_dis), min=0)

        # print("pos_exp:  ",pos_exp)
        # print("neg_exp:  ", neg_exp)
        with_pos_proxies=torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
                dim=1)  # The set of positive proxies of data in the batch
        # The number of positive proxies
        num_valid_proxies=len(with_pos_proxies)

        P_sim_sum=torch.where(P_one_hot == 1, pos_exp,
                              torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum=torch.where(N_one_hot == 1, neg_exp,
                              torch.zeros_like(neg_exp)).sum(dim=0)

        # pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        # neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

        pos_term=P_sim_sum.sum() / num_valid_proxies
        neg_term=N_sim_sum.sum() / self.nb_classes

        

        loss = pos_term + neg_term 

        return loss
      
class Adv_Rank_List_Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.4, m=1.4):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        #self.proxies = torch.nn.Parameter(proxies)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.m = m
        self.mrg = mrg

    def forward(self, X,):
        a, p, n = torch.chunk(X, 3, dim=0)
        #a = self.proxies

        ap_cos_dis = 1 - F.linear(l2_norm(p), l2_norm(a))
        an_cos_dis = 1 - F.linear(l2_norm(n), l2_norm(a))

        t = 20
        W = torch.exp(t * (self.m - an_cos_dis))
        W_sum0 = W.sum(dim=0)

        pos_exp = torch.clamp((ap_cos_dis - (self.m - self.mrg)), min=0)
        neg_exp = (W/W_sum0)*torch.clamp((self.m - an_cos_dis), min=0)

        P_sim_sum = pos_exp.sum(dim=0)
        N_sim_sum = neg_exp.sum(dim=0)

        pos_term = P_sim_sum.sum() / self.nb_classes
        neg_term = N_sim_sum.sum() / self.nb_classes

        loss = pos_term + neg_term

        return loss
    

class New_loss1(torch.nn.Module):
    def __init__(self, ):
        super(New_loss1, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss    
#My loss test
class New_loss_Origin(torch.nn.Module):

    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, m=1.4,lamda = 0.1):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        # print('loss is A')
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.m = m
        self.mrg = mrg
        self.alpha = alpha
        #########################
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        self.lamda = lamda
        ##################################

        #######new set
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        ###################################

    def forward(self, X, T,epoch=1):
        P = self.proxies

        # cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity

#######################################################
        hard_pairs = self.miner(X, T)
        loss_MS = self.loss_func(X, T, hard_pairs)
##########################################################
        cos = dist_matrix(l2_norm(X),l2_norm(P),c = 7) #c = 6
        # cos = dist_matrix(X,P,c = 6) #c = 6
        cos = abs(cos - 0.01)
        # cos_dis = 1-cos
        #####################################
        # cos_dis=abs(cos_dis-0.2)
        # cos_dis=cos_dis-0.1
        cos_dis = cos
        ####################################
        cos_dis_pos = cos_dis
        cos_dis_neg = cos_dis
#         cos_dis_pos = (cos_dis+1)*torch.log(1+cos_dis)
        
#         cos_dis_neg = (1-cos_dis)*torch.log(1-cos_dis)
        
#         cos_dis=torch.pow(cos_dis,1.1)
#         cos_dis=torch.pow(cos_dis,1.1)
        
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        #negtive pair weight
        # t = 4   #please set diffrent
        t=10

        W = torch.exp(t*(self.m - cos_dis_neg))
        W_sum = torch.where(N_one_hot == 1, W, torch.zeros_like(W)).sum(dim=0)
        W_sum = W_sum.sum()

        pos_exp = torch.exp(self.alpha * (2*cos_dis_pos - (self.m - self.mrg)))
        neg_exp = (W/W_sum)*torch.exp(self.alpha * (self.m - cos_dis_neg))
        
        # pos_exp = 
        # #pos_exp = torch.pow(pos_exp,1.2)
        # # #neg_exp = torch.pow(neg_exp,1.1)
        
        # pos_exp = torch.pow(pos_exp,1.2)
        # neg_exp = torch.pow(neg_exp,1.1)


        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes


        loss = pos_term + neg_term

        # temp_lamda = self.lamda*(epoch//5)
        temp_lamda = 0.2
        total_loss = (1-temp_lamda)*loss + temp_lamda*loss_MS
        return total_loss
    
##消融实验使用
#My loss test
class New_loss(torch.nn.Module):

    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, m=1.4,lamda = 0.1):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.m = m
        self.mrg = mrg
        self.alpha = alpha
        #########################
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        self.lamda = lamda
        ##################################

        #######new set
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        ###################################

    def forward(self, X, T,epoch=1):
        P = self.proxies
        hard_pairs = self.miner(X, T)
        loss_MS = self.loss_func(X, T, hard_pairs)
        cos = F.linear(l2_norm(X), l2_norm(P))
        # cos = dist_matrix(l2_norm(X),l2_norm(P),c = 7) #c = 6
        cos = abs(cos - 0.01)

        cos_dis = cos
        cos_dis_pos = cos_dis
        cos_dis_neg = cos_dis
        
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        t = 10
        W = torch.exp(t*(self.m - cos_dis_neg))
        W_sum = torch.where(N_one_hot == 1, W, torch.zeros_like(W)).sum(dim=0)
        W_sum = W_sum.sum()
        pos_exp = torch.exp(self.alpha * (2*cos_dis_pos - (self.m - self.mrg)))
        neg_exp = (W/W_sum)*torch.exp(self.alpha * (self.m - cos_dis_neg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  
        num_valid_proxies = len(with_pos_proxies) 
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term
        temp_lamda = 0.2
        total_loss = (1-temp_lamda)*loss #+ temp_lamda*loss_MS
        return loss_MS #total_loss

class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        # print(hard_pairs)
        # input()
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss( normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

class Circle(nn.Module):

    def __init__(self):
        super(Circle,self).__init__()
        self.loss_func = losses.CircleLoss()

    def forward(self, embeddings, labels):
        loss= self.loss_func(embeddings, labels)
        return loss

class GeneralizedLiftedStructureLoss(nn.Module):
    
    def __init__(self):
        super(GeneralizedLiftedStructureLoss,self).__init__()
        self.loss_func = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)

    def forward(self, embeddings, labels):
        loss= self.loss_func(embeddings, labels)
        return loss