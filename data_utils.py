'''
    数据处理代码
'''
from torch.utils.data import DataLoader,Dataset
import pandas as pd

class TextClassification(Dataset):
    def __init__(self, data_path):
        # 读取并去除异常样本
        self.data = pd.read_csv(data_path).dropna()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        review = self.data.iloc[idx]['review']
        label = self.data.iloc[idx]['label']
        return {'review': review, 'labels': label}





if __name__ == '__main__':
    data_path = './data/ChnSentiCorp_htl_all.csv'
    dataset = TextClassification(data_path)
