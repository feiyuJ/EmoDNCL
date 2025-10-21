"""

""" 
version = 1

import os 
import numpy as np 
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import scipy.io as sio
import glob  
from sklearn.model_selection import train_test_split


predictions_dir = './predictions'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)
             
subjects = 15 # Num. of subjects used for LOSO
classes = 4 # Num. of classes SEED:3, SEEDIV4:4

def to_categorical(y, num_classes=None, dtype='float32'): 
    #one-hot encoding
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0] 
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class EmotionDataset(InMemoryDataset):
    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, edge_index=None,
                 transform=None, pre_transform=None):
        self.stage = stage #Train or test
        self.subjects = subjects  
        self.sub_i = sub_i
        self.X = X
        self.Y = Y
        self.edge_index = edge_index
        
        #super(EmotionDataset, self).__init__(root, transform, pre_transform)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # 630个62节点的图数据，每个图数据有265*5个特征
        # import pdb;pdb.set_trace()

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['./V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                version, self.stage, self.subjects, self.sub_i)]
    def download(self):
        pass
    
    def process(self): 
        data_list = [] 
        # process by samples

        num_samples = np.shape(self.Y)[0]
        # print(num_samples)
        for sample_id in tqdm(range(num_samples)): 
            x = self.X[sample_id,:,:]    
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(self.Y[sample_id,:])
            data = Data(x=x, y=y)
            data_list.append(data) 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
def normalize(data):
    mee=np.mean(data,0)
    data=data-mee
    stdd=np.std(data,0)
    data=data/(stdd+1e-7)
    return data 

def get_data(data_path="./data/SEEDIV/data_100per/raw/"):
    """
    Load EEG emotion data from SEEDIV dataset
    
    Args:
        data_path (str): Path to the raw data directory
        
    Returns:
        tuple: (sub_mov, sub_label) - EEG data and labels
    """
    # Load labels for three sessions
    label1 = sio.loadmat(data_path+'label1.mat')['label']
    label2 = sio.loadmat(data_path+'label2.mat')['label']
    label3 = sio.loadmat(data_path+'label3.mat')['label']
    files = sorted(glob.glob(data_path+'*_*'))

    sublist = set()
    for f in files:
        sublist.add(f.split('/')[-1].split('_')[0] )
    
    print('Total number of subjects: {:.0f}'.format(len(sublist)))
    sublist = sorted(list(sublist))
    print(sublist)
   
    sub_mov = [] 
    sub_label = []
    
    for sub_i in range(subjects):   # 遍历15个受试者
        sub = sublist[sub_i]
        sub_files = glob.glob(path+sub+'_'+'*')  # 将每个受试者的3次实验数据路径存入sub_files
        # import pdb;pdb.set_trace()
        mov_data = [] 
        for f in sub_files:       # 将每个受试者的3次实验数据读取出来
            print(f)
            data = sio.loadmat(f, verify_compressed_data_integrity=True)
            keys = data.keys()
            de_mov = [k for k in keys if 'de_movingAve' in k]  # 读取单个受试者15个视频平均移动提取的微分熵数据
        
            mov_datai = [] 
            for t in range(len(de_mov)):   # 遍历15个视频
                temp_data = data[de_mov[t]].transpose(0,2,1)   
                # 取出单个受试者单个视频的微分熵数据(通道数，时间点，频带特征数)-->(通道数，频带特征数，时间点)
                data_length  = temp_data.shape[-1]
                # mov_i = np.zeros((62, 5, 265))  # 将所有数据同一到相同长度,不足部分用0填充 SEED
                mov_i = np.zeros((62, 5, 64))   # SEEDIV
                mov_i = np.resize(temp_data, (mov_i.shape[0], mov_i.shape[1], mov_i.shape[2]))  # 使用本身数据填充
                # mov_i[:,:,:data_length] = temp_data
                mov_i = mov_i.reshape(62, -1)#.transpose(1,0)
                
                mov_datai.append(mov_i) # 将单个受试者15个视频的数据存入mov_datai
            mov_datai = np.array(mov_datai)  # 将15个视频的数据转换为numpy数组(15, 62, 265*5)
            mov_data.append(mov_datai) # 将受试者3次的数据存入mov_data（15, 62, 265*5）*3
            
        mov_data = np.vstack(mov_data)   # 将单个人的三次实验数据合并(3*15, 62, 265*5)
        mov_data = normalize(mov_data)   # 数据归一化
        sub_mov.append(mov_data)         # 将所有受试者的数据存入sub_mov 15*(3*15, 62, 265*5)
        # sub_label.append(np.hstack([label, label, label]).squeeze())  # 将所有受试者的标签存入sub_label 15*(3*15,)只用一个session时
        # sub_label.append(np.hstack([label]).squeeze())
        sub_label.append(np.hstack([label1, label2, label3]).squeeze())   #SEEDIV
        
    sub_mov = np.array(sub_mov)   # (15, 45, 62, 265*5)   SEEDIV:(15, 72, 62, 64*5)
    sub_label = np.array(sub_label) # (15, 45)
    # import pdb;pdb.set_trace()

    return sub_mov, sub_label
    
def build_dataset(subjects, data_path="./data/SEEDIV/data_100per/raw/"):
    """
    Build dataset for cross-validation
    
    Args:
        subjects (int): Number of subjects
        data_path (str): Path to raw data directory
    """
    load_flag = True   # If data is already built, don't load again
    for sub_i in range(subjects):
        path = './data/SEEDIV/data_100per/processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                version, 'Train', subjects, sub_i)
        print(path)
        
        if not os.path.exists(path):
        
            if load_flag:  # Load data once
                mov_coefs, labels = get_data(data_path)
                # Load 15 subjects with 3 sessions each: mov_coefs:(15, 15*3, 62, 64*5), labels:(15, 15*3)
                used_coefs = mov_coefs
                load_flag = False
            
            index_list = list(range(subjects))  # 0-14，15个受试者
            del index_list[sub_i]     # 删除第i个受试者,作为测试集
            test_index = sub_i        # 第i个受试者作为测试集
            train_index = index_list  # 其他受试者作为训练集
            
            print('Building train and test dataset')
            #get train & test
            # X = used_coefs[train_index,:].reshape(-1, 62, 265*5)    # SEED
            X = used_coefs[train_index,:].reshape(-1, 62, 64*5)       # SEEDIV
            Y = labels[train_index,:].reshape(-1)
            # testX = used_coefs[test_index,:].reshape(-1, 62, 265*5)   # SEED
            testX = used_coefs[test_index,:].reshape(-1, 62, 64*5)   # SEEDIV
            testY = labels[test_index,:].reshape(-1) 
            #get labels
            _, Y = np.unique(Y, return_inverse=True)
            Y = to_categorical(Y, classes)#
            _, testY = np.unique(testY, return_inverse=True)
            testY = to_categorical(testY, classes)
            X_train_1per, _, y_train_1per, _ = train_test_split(X, Y, test_size=0.99,shuffle=True)
            # X_train_20per, _, y_train_20per, _ = train_test_split(X, Y, test_size=0.8,shuffle=True)
            # X_train_50per, _, y_train_50per, _ = train_test_split(X, Y, test_size=0.5,shuffle=True)
            # X_train_75per, _, y_train_75per, _ = train_test_split(X, Y, test_size=0.25,shuffle=True)
            X_train_100per, y_train_100per = X, Y
            # import pdb;pdb.set_trace()
            
            # train_dataset = EmotionDataset('Train', 'data/SEEDIV/data_0per/', subjects, sub_i, X, Y)
            # test_dataset = EmotionDataset('Test', 'data/SEEDIV/data_0per/', subjects, sub_i, testX, testY)
            train_1per_dataset = EmotionDataset('Train', 'SEEDIV/data_1per/', subjects, sub_i, X_train_1per, y_train_1per)
            test_1per_dataset = EmotionDataset('Test', 'SEEDIV/data_1per/', subjects, sub_i, testX, testY)
            # train_20per_dataset = EmotionDataset('Train', 'SEEDIV/data_20per/', subjects, sub_i, X_train_20per, y_train_20per)
            # test_20per_dataset = EmotionDataset('Test', 'SEEDIV/data_20per/', subjects, sub_i, testX, testY)
            # train_50per_dataset = EmotionDataset('Train', 'SEEDIV/data_50per/', subjects, sub_i, X_train_50per, y_train_50per)
            # test_50per_dataset = EmotionDataset('Test', 'SEEDIV/data_50per/', subjects, sub_i, testX, testY)
            # train_75per_dataset = EmotionDataset('Train', 'SEEDIV/data_75per/', subjects, sub_i, X_train_75per, y_train_75per)
            # test_75per_dataset = EmotionDataset('Test', 'SEEDIV/data_75per/', subjects, sub_i, testX, testY)
            train_100per_dataset = EmotionDataset('Train', 'SEEDIV/data_100per/', subjects, sub_i, X_train_100per, y_train_100per)
            test_100per_dataset = EmotionDataset('Test', 'SEEDIV/data_100per/', subjects, sub_i, testX, testY)

            print('Dataset is built.')
            
def get_dataset(subjects, sub_i, Dataset, data_per):
    """
    Get train and test datasets for cross-validation
    
    Args:
        subjects (int): Number of subjects
        sub_i (int): Subject index for testing
        Dataset (str): Dataset name (SEED or SEEDIV)
        data_per (int): Data percentage (1, 5, 10, 50, 75, 100)
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    path = './data/{}/data_{}per/processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
        Dataset, data_per, version, 'Train', subjects, sub_i)
    print(path)
    if not os.path.exists(path): 
        raise IOError('Train dataset does not exist! Please run build_dataset() first.')
    
    train_dataset = EmotionDataset('Train', f'./data/{Dataset}/data_{data_per}per', subjects, sub_i)
    test_dataset = EmotionDataset('Test', f'./data/{Dataset}/data_{data_per}per', subjects, sub_i)

    return train_dataset, test_dataset


if __name__ == '__main__':
    build_dataset(subjects)# Build dataset for each fold
