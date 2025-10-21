"""
Data Loading and Preprocessing for EEG Datasets

Supports:
- SEED dataset (3 emotion classes)
- SEEDIV dataset (4 emotion classes)

Functions:
- get_dataset: Load train/test datasets for cross-validation
- build_dataset: Process raw data into PyTorch Geometric format
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
    output_shape = input_shape + (num_classes, )
    categorical = np.reshape(categorical, output_shape)
    return categorical

class EEGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(EEGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in range(len(self.raw_paths)):
            data = torch.load(self.raw_paths[i])
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def build_dataset(data_path, subjects, cv_n, data_per=100):
    """Build dataset for cross-validation
    
    Args:
        data_path: Path to dataset files
        subjects: Number of subjects
        cv_n: Cross-validation fold number
        data_per: Percentage of data to use (100 for full dataset)
        
    Returns:
        train_dataset, test_dataset: PyTorch Geometric datasets
    """
    train_data_list = []
    test_data_list = []
    
    for subject in range(subjects):
        if subject == cv_n:
            # Test subject
            file_path = os.path.join(data_path, f'V_1_Train_CV15_{subject}.dataset')
            if os.path.exists(file_path):
                data = torch.load(file_path)
                test_data_list.append(data)
        else:
            # Train subjects
            file_path = os.path.join(data_path, f'V_1_Train_CV15_{subject}.dataset')
            if os.path.exists(file_path):
                data = torch.load(file_path)
                train_data_list.append(data)
    
    # Create datasets
    train_dataset = EEGDataset(root='./temp_train', transform=None, pre_transform=None)
    train_dataset.data, train_dataset.slices = train_data_list, None
    
    test_dataset = EEGDataset(root='./temp_test', transform=None, pre_transform=None)
    test_dataset.data, test_dataset.slices = test_data_list, None
    
    return train_dataset, test_dataset

def get_dataset(subjects, cv_n, dataset_name, data_per=100):
    """Get train and test datasets for cross-validation
    
    Args:
        subjects: Number of subjects
        cv_n: Cross-validation fold number
        dataset_name: Name of dataset ('SEED' or 'SEEDIV')
        data_per: Percentage of data to use
        
    Returns:
        train_dataset, test_dataset: PyTorch Geometric datasets
    """
    if dataset_name == 'SEED':
        data_path = './data/SEED/data_100per/processed'
    elif dataset_name == 'SEEDIV':
        data_path = './data/SEEDIV/data_100per/processed'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return build_dataset(data_path, subjects, cv_n, data_per)
