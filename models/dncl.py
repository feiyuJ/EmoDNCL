# -*- coding: utf-8 -*-
"""
DNCL: Deep Neural Contrastive Learning for EEG Emotion Recognition

This module implements the DNCL model combining:
- Self-distillation mechanism (originally DINO-inspired)
- SOGNN (Self-Organized Graph Neural Network)

Main Components:
- SOGC: Self-Organized Graph Convolution layer
- Network: Base feature extraction network
- DNCL: Main model with teacher-student architecture

Usage:
    from models.dncl import DNCL, SOGNN
    model = DNCL(SOGNN, dataset='SEEDIV', train_model='self_supervised')
"""
import torch.nn as nn
import numpy as np
import math
import torch 
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GraphConv, DenseSAGEConv, dense_diff_pool, DenseGCNConv,GATConv
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling
from torch.nn import Linear, Dropout, PReLU, Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse


class SOGC(torch.nn.Module):
    """Self-organized Graph Construction Module
    
    Args:
        in_features: size of each input sample
        bn_features: size of bottleneck layer
        out_features: size of each output sample
        topk: size of top k-largest connections of each channel
    """
    def __init__(self, in_features: int, bn_features: int, out_features: int, topk: int):
        super().__init__()

        self.channels = 62
        self.in_features = in_features
        self.bn_features = bn_features
        self.out_features = out_features
        self.topk = topk
        
        self.bnlin = Linear(in_features, bn_features)
        self.gconv = DenseGCNConv(in_features, out_features)

    def forward(self, x):
        x = x.reshape(-1, self.channels, self.in_features)
        xa = torch.tanh(self.bnlin(x))
        adj = torch.matmul(xa, xa.transpose(2,1))
        adj = torch.softmax(adj, 2)

        amask = torch.zeros(xa.size(0), self.channels, self.channels).to(x.device)
        amask.fill_(0.0)
        for i in range(xa.size(0)):
            _, top_indices = torch.topk(adj[i], self.topk, dim=1)
            amask[i].scatter_(1, top_indices, 1)
        
        adj = adj * amask
        adj = adj + torch.eye(self.channels).to(x.device)
        
        x = self.gconv(x, adj)
        return x

class Network(nn.Module):
    """Base feature extraction network for EEG emotion recognition
    
    Args:
        data_set: Dataset name ('SEED' or 'SEEDIV')
        model: Training mode ('self_supervised' or 'supervised')
    """
    def __init__(self, data_set, model='self_supervised'):
        super().__init__()
        self.data_set = data_set
        if data_set == 'SEED':
            pool_size = 4
            pool_size_last = 3
            num_classes = 3
        elif data_set == 'SEEDIV':
            pool_size = 2
            pool_size_last = 2
            num_classes = 4
        drop_rate = 0.05
        self.train_model = model
        topk = 62
        self.channels = 62 
        
        self.conv1 = Conv2d(1, 32, (5,5))
        self.drop1 = Dropout(drop_rate)
        self.pool1 = MaxPool2d((1, pool_size))
        if data_set == 'SEEDIV':
            self.sogc1 = SOGC(30*32, 64, 32, topk)
        elif data_set == 'SEED':
            self.sogc1 = SOGC(15*32, 64, 32, topk)
        
        self.sogc2 = SOGC(32, 64, 32, topk)
        self.sogc3 = SOGC(32, 64, 32, topk)
        
        self.pool2 = MaxPool2d((1, pool_size_last))
        self.drop2 = Dropout(drop_rate)
        
        if data_set == 'SEEDIV':
            self.fc = Linear(2*32, num_classes)
        elif data_set == 'SEED':
            self.fc = Linear(3*32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)
        
        if self.data_set == 'SEEDIV':
            x = x.reshape(x.size(0), -1, 30*32)
        elif self.data_set == 'SEED':
            x = x.reshape(x.size(0), -1, 15*32)
        
        x = self.sogc1(x)
        x = self.sogc2(x)
        x = self.sogc3(x)
        
        x = self.pool2(x)
        x = self.drop2(x)
        
        if self.data_set == 'SEEDIV':
            x = x.reshape(x.size(0), -1, 2*32)
        elif self.data_set == 'SEED':
            x = x.reshape(x.size(0), -1, 3*32)
        
        x = self.fc(x)
        return x

class SOGNN(nn.Module):
    """Self-Organized Graph Neural Network
    
    Alias for Network class for backward compatibility
    """
    def __init__(self, data_set, model='self_supervised'):
        super().__init__()
        self.network = Network(data_set, model)
    
    def forward(self, x):
        return self.network(x)

class DNCL(nn.Module):
    """DNCL: Deep Neural Contrastive Learning model
    
    Teacher-student self-distillation framework for EEG emotion recognition.
    
    Args:
        model: Base network architecture
        data_set: Dataset name ('SEED' or 'SEEDIV')
        tps: Student temperature parameter
        tpt: Teacher temperature parameter
        m: Teacher momentum coefficient
        out_dim: Output dimension
        momentum: Center momentum coefficient
        use_soft_CL: Whether to use soft contrastive learning
        train_model: Training mode ('self_supervised' or 'FT')
        device: Device to run on
    """
    def __init__(self, model, data_set, tps=0.1, tpt=0.03, m=0.9, out_dim=62, momentum=0.9, 
                 use_soft_CL=True, train_model='self_supervised', device=None):
        super(DNCL, self).__init__()
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.center = torch.zeros(1, out_dim).to(self.device)
        self.tps = tps
        self.tpt = tpt
        self.m = m
        self.momentum = momentum
        self.use_soft_CL = use_soft_CL
        self.train_model = train_model

        # Initialize student and teacher networks
        self.student = model(data_set, train_model)
        self.teacher = model(data_set, train_model)

        # Initialize teacher weights with student weights and freeze teacher
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

        # Initialize queue for contrastive learning
        self.register_buffer("queue", torch.randn(62, 1024))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def to(self, device):
        """Override to method to ensure self.center moves to correct device"""
        super().to(device)
        self.center = self.center.to(device)
        return self

    @torch.no_grad()
    def _update_teacher_network(self):
        """Update teacher network with EMA"""
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(self.m).add_(param_s.data, alpha=1 - self.m)

    @torch.no_grad()
    def _update_center(self, batch_center):
        """Update center with momentum
        
        Args:
            batch_center: Current batch center
        """
        if not hasattr(self, 'center_initialized'):
            self.center_initialized = False

        if self.center_initialized:
            self.center = self.momentum * self.center + (1 - self.momentum) * batch_center
        else:
            self.center = batch_center
            self.center_initialized = True

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size <= self.queue.shape[1]:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue.shape[1]
            self.queue_ptr[0] = ptr
        else:
            self.queue[:, ptr:] = keys.T[:, :self.queue.shape[1] - ptr]
            remaining = batch_size - (self.queue.shape[1] - ptr)
            self.queue[:, :remaining] = keys.T[:, self.queue.shape[1] - ptr:]
            ptr = remaining
            self.queue_ptr[0] = ptr

    def H(self, t, s):
        """Compute self-distillation loss between teacher and student
        
        Args:
            t: Teacher network output (detached)
            s: Student network output
            
        Returns:
            Distillation loss (mean cross-entropy)
        """
        t = t.detach()
        s = F.softmax(s / self.tps, dim=1)
        t_centered = F.softmax((t - self.center) / self.tpt, dim=1)
        loss = -(t_centered * torch.log(s)).sum(dim=1).mean()
        return loss

    def H_instudent(self, s1, s2):
        """Compute self-distillation loss between student outputs
        
        Args:
            s1: Student network output 1
            s2: Student network output 2
            
        Returns:
            Internal distillation loss (mean cross-entropy)
        """
        s1 = F.softmax(s1 / self.tps, dim=1)
        s2 = F.softmax(s2 / self.tps, dim=1)
        loss = -(s2 * torch.log(s1)).sum(dim=1).mean()
        return loss

    def forward(self, im_q, im_k):
        """Forward pass
        
        Args:
            im_q: Input query (augmented version 1)
            im_k: Input key (augmented version 2)
            
        Returns:
            Loss dictionary, features, and adjacency matrix
        """
        if self.train_model == 'FT':
            features_t1, pred, logits, matx = self.teacher(im_q)
            return features_t1, pred, logits, matx

        elif self.train_model == 'self_supervised':
            features_s1, pred, _, matx1 = self.student(im_q)
            query_s1 = nn.functional.normalize(features_s1, dim=1)

            im_k2 = im_k.clone()
            features_s2, pred, _, matx2 = self.student(im_k)
            query_s2 = nn.functional.normalize(features_s2, dim=1)

            features_t2, pred, _, matx3 = self.teacher(im_k2)
            query_t2 = nn.functional.normalize(features_t2, dim=1)

            self._update_teacher_network()
            matx = matx1 + matx2 + matx3
            loss = (2.5*self.H(query_t2, query_s1) + 2.5*self.H_instudent(query_s1, query_s2))/5
            return {'loss': loss}, features_s1, features_s2, features_t2, matx
