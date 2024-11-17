import numpy as np
import torch
import logging
import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math
from scipy.special import comb

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from hyptorch.pmath import dist_matrix

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))

def acc_small(T, Y, t1):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t1 in torch.Tensor(y).long()[0] and t1==t:
            s += 1
    return s

def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    print('---------------------------------------------')
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def evaluate_small(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    small_num = acc_small(T, Y,3)
    print("acc_small : {:.3f}".format(small_num))
    return T

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
            
        return match_counter / m
    
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
                
    return recall

def evaluate_cos_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            y = T[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall
  
def evaluate_acc_mae_f1(model, dataloader):
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 1 neighbors with cosine
    K = 32
    Y = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    #Y = Y.float().cpu()
    Y = Y.long()[:,0]
    # print('Y: ',Y,Y.shape)
    # print('T: ',T,T.shape)

    acc = metrics.accuracy_score(T, Y)
    mae = metrics.mean_absolute_error(T, Y)
    f1 = metrics.f1_score(T, Y, average='macro')
    print('----------------------------------------------------')
    print('acc: ',acc)
    print('MAE: ',mae)
    print('F1: ',f1)
    

    return acc, mae, f1
  
def evaluate_acc(model, dataloader):
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 1 neighbors with cosine
    K = 32
    Y = []
    
    cos_sim = F.linear(X, X)
    # cos_sim = dist_matrix(X,X,c = 6)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    #Y = Y.float().cpu()
    Y = Y[:,0]

    acc = metrics.accuracy_score(T, Y)
    print("\nACC : {:.3f}".format(100 * acc))

    return acc
    
    #用来测试绝对误差
def absolute_acc(model, dataloader):
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 1 neighbors with cosine
    K = 32
    Y = []
    
    cos_sim = F.linear(X, X)
    # cos_sim = dist_matrix(X,X,c = 6)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    #Y = Y.float().cpu()
    Y = Y[:,0]
    acc = abs(sum(Y-T))/T.shape[0]
    print("\nACC : {:.3f}".format(100 * acc))
    return acc
  
def evaluate_mae(model, dataloader):
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 1 neighbors with cosine
    K = 32
    Y = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    #Y = Y.float().cpu()
    Y = Y[:,0]

    mae = metrics.mean_absolute_error(T, Y)
    print("\nMAE : {:.3f}".format(mae))
    
    return mae
def evaluate_confusion(model, dataloader):
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 1 neighbors with cosine
    K = 32
    Y = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    #Y = Y.float().cpu()
    Y = Y[:,0]

    confusion = metrics.confusion_matrix(T, Y)
    # acc = metrics.accuracy_score(T,Y)
    # print("\nACC : {:.3f}".format(100 * acc))
    acc = metrics.accuracy_score(T, Y)
    print("\nACC : {:.3f}".format(100 * acc))

    return confusion,acc

def evaluate_f1(model, dataloader):
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 1 neighbors with cosine
    K = 32
    Y = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    #Y = Y.float().cpu()
    Y = Y[:,0]

    f1 = metrics.f1_score(T, Y, average='weighted')
    print("\nF1 score : {:.3f}".format(100 * f1))

    return f1
  
def evaluate_cluster(model, dataloader):
    n_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    features, labels = predict_batchwise(model, dataloader)
    features = l2_norm(features).cpu().numpy()
    # k-means algorithms
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(features)
    centers = kmeans.cluster_centers_

    # k-NN algorithms
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(centers, range(len(centers)))

    idx_feat = neigh.predict(features)
    nums = len(features)
    ds = np.zeros(nums)
    for i in range(nums):
        ds[i] = np.linalg.norm(features[i, :] - centers[idx_feat[i], :])

    labels_pre = np.zeros(nums)
    for i in range(n_classes):
        idx = np.where(idx_feat == i)[0]
        ind = np.argmin(ds[idx])
        cid = idx[ind]
        labels_pre[idx] = cid

    NMI, F1 = compute_cluster_metric(labels.cpu().numpy(), labels_pre)
    return NMI, F1


def compute_cluster_metric(labels, labels_pre):
    N = len(labels)
    centers = np.unique(labels)
    n_clusters = len(centers)

    # count the number
    count_cluster = np.zeros(n_clusters)
    for i in range(n_clusters):
        count_cluster[i] = len(np.where(labels == centers[i])[0])

    # map labels_pre into item_map
    keys = np.unique(labels_pre)
    nums_item = len(keys)
    values = range(nums_item)
    item_map = dict()
    for i in range(nums_item):
        item_map[keys[i]] = values[i]
        # item_map.update([keys[i], values[i]])

    # count the number
    count_item = np.zeros(nums_item)
    for i in range(N):
        idx = item_map[labels_pre[i]]
        count_item[idx] += 1

    # compute purity
    purity = 0
    for i in range(n_clusters):
        member = np.where(labels == centers[i])[0]
        member_ids = labels_pre[member]

        count = np.zeros(nums_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1
        purity = purity + max(count)

    purity = purity / N

    # compute Normalized Mutual Information (NMI)
    count_cross = np.zeros((n_clusters, nums_item))
    for i in range(N):
        index_cluster = np.where(labels[i] == centers)[0]
        index_item = item_map[labels_pre[i]]
        count_cross[index_cluster, index_item] += 1

    I = 0
    for k in range(n_clusters):
        for j in range(nums_item):
            if count_cross[k, j] > 0:
                s = count_cross[k, j] / N * math.log(N * count_cross[k, j] / (count_cluster[k] * count_item[j]))
                I = I + s

    # entropy
    H_cluster = 0
    for k in range(n_clusters):
        s = -count_cluster[k] / N * math.log(count_cluster[k] / float(N))
        H_cluster = H_cluster + s

    H_item = 0
    for j in range(nums_item):
        s = -count_item[j] / N * math.log(count_item[j] / float(N))
        H_item = H_item + s

    NMI = 2 * I / (H_cluster + H_item)

    # computer F-measure
    tp_fp = 0
    for k in range(n_clusters):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    # compute True Positive (TP)
    tp = 0
    for k in range(n_clusters):
        member = np.where(labels == centers[k])[0]
        member_ids = labels_pre[member]

        count = np.zeros(nums_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(nums_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    # False Positive (FP)
    fp = tp_fp - tp

    # compute False Negative (FN)
    count = 0
    for j in range(nums_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)

    fn = count - tp

    # compute True Negative (TN)
    tn = N * (N - 1) / 2 - tp - fp - fn

    # compute RI
    RI = (tp + tn) / (tp + fp + fn + tn)

    # compute F measure
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    beta = 1
    F1 = (beta * beta + 1) * P * R / (beta * beta * P + R)

    return NMI, F1
