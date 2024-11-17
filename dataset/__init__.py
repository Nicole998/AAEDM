from .kongyu import KongYu
from .import utils
from .base import BaseDataset
from .kybase import MyDataset


_type = {

    'kongyu': KongYu
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
