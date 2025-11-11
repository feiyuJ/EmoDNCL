# -*- coding: utf-8 -*-
"""
Data Augmentation for EEG Signals

NINA: Neighbor-based Intelligent Neural Augmentation for EEG signals
"""
import numpy as np
import torch
import random
import pandas as pd

def load_neighbor_info(file_path):
    """
    加载邻接关系文件，并转换为字典格式。
    参数:
    file_path (str): 邻接关系文件路径 (CSV 格式)。

    返回:
    dict: 邻接关系字典，格式为 {channel: [neighbor_channels]}。
    """
    df = pd.read_csv(file_path)
    neighbor_info = {str(row["Channel"]): row["Neighbors"].split(",") for _, row in df.iterrows()}
    return neighbor_info

def NINA(eegdata, n, p, t, random_seed=None):
    """
    NINA: Neighbor-based Intelligent Neural Augmentation for EEG signals
    
    参数:
    eegdata (torch.Tensor): EEG 信号，形状为 (batch_size, channels, time_steps)。
    n (int): 要增强的通道数量 (0 <= n <= 总通道数)。
    p (float): 修改比例：
               - p = 1: 用邻域的最大值或最小值替换。
               - p = 0.5: 替换为原始值与最大值或最小值的平均值。
    t (float): 时间维度上参与增强的比例 (0 <= t <= 1)。
    random_seed (int, optional): 随机种子，确保结果可复现，一般不使用。
    
    返回:
    torch.Tensor: 增强后的 EEG 信号。
    """
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    if eegdata.dim() != 3:
        raise ValueError(f"输入的张量必须是三维的 (batch_size, channels, time_steps)，但得到的是 {eegdata.dim()} 维。")

    device = eegdata.device
    eegdata = eegdata.clone().to(device)
    batch_size, channels, time_steps = eegdata.size()

    neighbor_info = load_neighbor_info('./datasets/neighbors_fixed_full.csv')

    if not (0 <= n <= channels):
        raise ValueError(f"n 必须在 0 和 {channels} 之间，但得到的是 {n}。")
    if not (0 <= t <= 1):
        raise ValueError(f"t 必须在 0 和 1 之间，但得到的是 {t}。")

    # 随机选择要增强的通道
    all_channels = list(range(channels))
    random.shuffle(all_channels)
    selected_channels = all_channels[:n]

    num_time_steps_to_augment = int(time_steps * t)
    time_indices_to_augment = random.sample(range(time_steps), num_time_steps_to_augment)

    augmented_data = eegdata.clone()

    for channel in selected_channels:
        neighbor_channels = neighbor_info.get(str(channel), [])
        neighbor_channels = [int(ch) for ch in neighbor_channels if ch.strip().isdigit()]

        if not neighbor_channels:
            continue

        neighbor_values = eegdata[:, neighbor_channels, :]
        max_values, _ = neighbor_values.max(dim=1)
        min_values, _ = neighbor_values.min(dim=1)

        new_values = max_values if random.choice(["max", "min"]) == "max" else min_values

        mask = torch.zeros_like(augmented_data[:, channel, :], dtype=torch.bool, device=device)
        mask[:, time_indices_to_augment] = True
        augmented_data[:, channel, :] = torch.where(
            mask,
            (1 - p) * eegdata[:, channel, :] + p * new_values,
            augmented_data[:, channel, :]
        )

    return augmented_data
