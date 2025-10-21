"""
DNCL Training Script

Two-stage training:
1. Self-supervised pre-training
2. Fine-tuning with labels

Usage:
    # Pre-training
    python main.py --training_mode self_supervised --selected_dataset SEEDIV --epochs 400 --gpu_id 0
    
    # Fine-tuning
    python main.py --training_mode FT --selected_dataset SEEDIV --epochs 300 --gpu_id 0
"""
import random
from data.augmentation import NINA
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
import os
import argparse
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F
import scipy.io as sio
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging

from data.loader import build_dataset, get_dataset
from models.dncl import SOGNN, DNCL
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--experiment_description', default='DNCL_Experiment', type=str,
                    help='Experiment Description')
parser.add_argument('--classes', default=4, type=int)
parser.add_argument('--training_mode', default='FT', type=str,
                    help='Modes of choice: self_supervised, FT')
parser.add_argument('--selected_dataset', default='SEEDIV', type=str,
                    help='Dataset of choice: SEED, SEEDIV')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--gpu_id', default=0, type=int,
                    help='GPU ID to use')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--training_per', default=100, type=int, help='[1, 5, 10, 50, 75, 100]')
parser.add_argument('--epochs', default=300, type=int)

parser.add_argument('--aug1_n', default=9, type=int)
parser.add_argument('--aug2_n', default=6, type=int)
parser.add_argument('--aug1_r', default=0.3, type=float, help='Augmentation time ratio')
parser.add_argument('--aug2_r', default=0.3, type=float, help='Augmentation time ratio')
parser.add_argument('--aug1_p', default=0.5, type=float, help='Augmentation replacement ratio')
parser.add_argument('--aug2_p', default=0.5, type=float, help='Augmentation replacement ratio')
parser.add_argument('--seednum', default=42, type=int, help='Random seed')

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seednum)
np.random.seed(args.seednum)
random.seed(args.seednum)

# Parameters
subjects = 15
Batch_size = args.batch_size
device = args.device
Dataset = args.selected_dataset
session = args.experiment_description

# Hyperparameters
cltps = 0.1
cltpt = 0.03
clm = 0.9
clout_dim = 62
clmomentum = 0.9
train_model = args.training_mode

if args.selected_dataset == 'SEED':
    classes = 3
elif args.selected_dataset == 'SEEDIV':
    classes = 4

Network = SOGNN

def train(model, train_loader, crit, optimizer, sensitive_channels):
    """Training function"""
    model.train()
    loss_all = 0

    num_batches = len(train_loader)
    batch_size = 16
    all_matrices = torch.zeros(num_batches, batch_size, 62, 62).to(device)

    for i, data in enumerate(train_loader):
        data = data.to(device)
        data_in, _ = to_dense_batch(data.x, data.batch)

        # Data augmentation using NINA method
        aug1 = NINA(data_in.clone(), n=args.aug1_n, p=args.aug1_p, t=args.aug1_r, random_seed=None)
        aug2 = NINA(data_in.clone(), n=args.aug2_n, p=args.aug2_p, t=args.aug2_r, random_seed=None)

        optimizer.zero_grad()

        # Forward pass
        output, features_s1, features_s2, features_t2, matx = model(aug1, aug2)
        loss = output['loss']
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

        # Store adjacency matrix
        all_matrices[i] = matx.detach().cpu()

    avg_mat_epoch = all_matrices.mean(dim=0)
    return loss_all / len(train_loader), avg_mat_epoch

def evaluate(model, loader, classes, device):
    """Evaluation function"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, classes)
            data = data.to(device)
            data_in, _ = to_dense_batch(data.x, data.batch)

            if train_model in ['self_supervised']:
                output, _, _, _, _ = model(data_in, data_in)
                pred = output['loss']
            else:
                output, features_s1, features_s2, features_t2, matx = model(data_in, data_in)
                pred = features_s1

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(all_preds.shape) > 1 and all_preds.shape[1] > 1:
        pred_labels = np.argmax(all_preds, axis=1)
        true_labels = np.argmax(all_labels, axis=1)
    else:
        pred_labels = all_preds
        true_labels = all_labels

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    try:
        auc = roc_auc_score(true_labels, all_preds, multi_class='ovr', average='weighted')
    except:
        auc = 0.0

    return auc, acc, f1

def main():
    """Main function"""
    print("=" * 60)
    print(f"DNCL Training - {train_model} mode")
    print(f"Dataset: {Dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"GPU: {args.gpu_id}")
    print("=" * 60)

    # Cross validation
    for cv_n in range(0, subjects):
        print(f"Cross Validation {cv_n}/{subjects}")
        
        # Load data
        train_dataset, test_dataset = get_dataset(subjects, cv_n, Dataset, data_per=args.training_per)
        train_loader = DataLoader(train_dataset, batch_size=Batch_size, drop_last=False, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=Batch_size)

        # Device setup
        device = torch.device('cuda', args.gpu_id) if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using device: {device}")

        # Create model
        model = DNCL(Network, Dataset, tps=cltps, tpt=cltpt, m=clm, out_dim=clout_dim, 
                    momentum=clmomentum, train_model=train_model, use_soft_CL=True, device=device).to(device)

        # Load pre-trained model for fine-tuning
        if train_model in ['fine_tune', 'FT']:
            load_from = f'./model/{Dataset}/{session}/self_supervised/{cv_n}/saved_models/ckp_last.pt'
            if os.path.exists(load_from):
                checkpoint = torch.load(load_from, map_location=device)
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                del_list = ['logits']
                pretrained_dict_copy = pretrained_dict.copy()
                for i in pretrained_dict_copy.keys():
                    if i in del_list:
                        del pretrained_dict[i]
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)
        crit = torch.nn.CrossEntropyLoss()

        # Training loop
        best_auc = 0
        best_acc = 0
        best_f1 = 0
        
        for epoch in range(args.epochs):
            start_time = time.time()
            
            if train_model in ['self_supervised']:
                loss, avg_mat_epoch = train(model, train_loader, crit, optimizer, None)
                val_auc, val_acc, val_f1 = evaluate(model, test_loader, classes, device)
            else:
                # Supervised learning mode
                model.train()
                loss_all = 0
                for data in train_loader:
                    data = data.to(device)
                    data_in, _ = to_dense_batch(data.x, data.batch)
                    optimizer.zero_grad()
                    label = torch.argmax(data.y.view(-1, classes), axis=1)
                    label = label.to(device)
                    output, features_s1, features_s2, features_t2, matx = model(data_in, data_in)
                    loss = crit(features_s1, label)
                    loss.backward()
                    loss_all += data.num_graphs * loss.item()
                    optimizer.step()
                
                val_auc, val_acc, val_f1 = evaluate(model, test_loader, classes, device)
                loss = loss_all / len(train_loader)

            epoch_time = time.time() - start_time
            
            # Record best results
            if val_auc > best_auc:
                best_auc = val_auc
                best_acc = val_acc
                best_f1 = val_f1

            # Print results
            print(f"V{cv_n}, EP{epoch+1:03d}, Loss:{loss:.3f}, "
                  f"AUC:{val_auc:.5f}, Acc:{val_acc:.5f}, "
                  f"VAUC:{best_auc:.5f}, Vacc:{best_acc:.5f}, Time: {epoch_time:.2f}")

            # Early stopping (optional)
            if epoch > 50 and val_auc < 0.5:
                break

        # Save results
        result_dir = f"./result/{Dataset}/{session}"
        os.makedirs(result_dir, exist_ok=True)
        
        result = {
            'cv': cv_n,
            'best_auc': best_auc,
            'best_acc': best_acc,
            'best_f1': best_f1,
            'epochs': epoch + 1
        }
        
        result_df = pd.DataFrame([result])
        result_file = f"{result_dir}/results_cv{cv_n}.csv"
        result_df.to_csv(result_file, index=False)
        
        print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
