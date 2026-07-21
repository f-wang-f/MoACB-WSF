import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import MinMaxScaler

from .config import (
    DATA_FILENAME, FEATURE_COLUMNS, TARGET_COLUMN, SEQUENCE_LENGTH,
    TRAIN_RATIO, VAL_RATIO, MIN_SAMPLES
)


class WindDataset(Dataset):
    def __init__(self, data, sequence_length=SEQUENCE_LENGTH):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, -1]
        return torch.FloatTensor(x), torch.FloatTensor([y])


def load_and_preprocess_data(filename=DATA_FILENAME):
    print('正在读取风速数据...')
    try:
        data = pd.read_excel(filename)
        for col in FEATURE_COLUMNS:
            if col not in data.columns:
                raise ValueError(f'未找到特征列: {col}')
        feature_data = data[FEATURE_COLUMNS].values
    except Exception as e:
        print(f'无法读取 Excel 文件: {e}')
        raise

    valid_rows = np.all(~np.isnan(feature_data) & ~np.isinf(feature_data), axis=1)
    feature_data = feature_data[valid_rows, :]
    if feature_data.shape[0] < MIN_SAMPLES:
        raise ValueError(f'数据不足(少于{MIN_SAMPLES}个样本),请提供更多数据。')

    scaler = MinMaxScaler()
    feature_data_norm = scaler.fit_transform(feature_data)
    min_vals = scaler.data_min_
    max_vals = scaler.data_max_

    dataset = WindDataset(feature_data_norm, SEQUENCE_LENGTH)

    num_samples = len(dataset)
    num_train = int(TRAIN_RATIO * num_samples)
    num_val = int(VAL_RATIO * num_samples)
    num_test = num_samples - num_train - num_val

    train_dataset = Subset(dataset, range(num_train))
    val_dataset = Subset(dataset, range(num_train, num_train + num_val))
    test_dataset = Subset(dataset, range(num_train + num_val, num_samples))

    min_speed = min_vals[-1]
    max_speed = max_vals[-1]
    
    print(f'数据预处理完成。训练样本数:{num_train},验证样本数:{num_val},测试样本数:{num_test}')

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'min_speed': min_speed,
        'max_speed': max_speed,
        'num_features': len(FEATURE_COLUMNS),
        'feature_columns': FEATURE_COLUMNS
    }