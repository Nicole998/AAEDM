from .kybase import *

class KongYu(MyDataset):
    def __init__(self, root,  mode, transform = None):
        self.data =  '/root/autodl-tmp/data/201116-X_all(label_noise_remove_0.3,size3605).npy'
        self.labels = '/root/autodl-tmp/data/201116-Y_all(label_noise_remove_0.3,size3605).npy'
        self.mode = mode
        self.transform = transform
    
        
        MyDataset.__init__(self, self.data, self.labels, self.mode, self.transform)
        