import os.path as osp 
import torch 
from torch_geometric.data import Dataset, download_url 
class MyDataset(Dataset): 
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyDataset, self).__init__(root, transform, pre_transform)
    def raw_file_names(self) -> Union[str, List[str], Tuple]: 
        return ['file_1', 'file_2', ...] 
    def processed_file_names(self) -> Union[str, List[str], Tuple]: 
        return ['data_1.pt', ...] 
    def download(self): 
        path = download_url(url, self.raw_dir) 
        def process(self): i = 0 
        for raw_path in self.raw_paths: # 读取数据 data = Data(...) 
            # 过滤数据集 
            if self.pre_filter is not None and not self.pre_filter(data):
            if self.pre_transform is not None: 
                data = self.pre_transform(data) 
                # 保存数据 torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i))) i += 1 
    def len(self): 
        return len(self.processed_file_names) 
        def get(self,idx):
            data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx))) 
            return data