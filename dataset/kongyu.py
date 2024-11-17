from .kybase import *

class KongYu(MyDataset):
    def __init__(self, root,  mode, transform = None):
        self.data =  '/root/autodl-tmp/data/data.npy'
        self.labels = '/root/autodl-tmp/data/lable.npy'
        self.mode = mode
        self.transform = transform
    
        
        MyDataset.__init__(self, self.data, self.labels, self.mode, self.transform)
        
