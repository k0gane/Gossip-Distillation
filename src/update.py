import torch
from torch.utils.data import Dataset

# 各エポックごとに行う処理が入ってるやつ

class DatasetSplit(Dataset):
    """
    Pytorch Dataset クラスをラップした抽象 Dataset クラスです。
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        tmp = []
        for idx in self.idxs:
            tmp.append(self.dataset[idx][1])
        print(tmp)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)