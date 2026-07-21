import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import traceback

from .config import EVAL_EPOCHS, EVAL_PATIENCE, LAMBDA_REG, GRADIENT_CLIP, MAX_MODEL_SIZE
from .model import HybridCNNBiLSTM
from .encoding import decode_individual, decode_hyperparams


def evaluate_individual(individual, train_dataset, val_dataset, min_speed, max_speed, num_features, sequence_length,
                        device, epochs=EVAL_EPOCHS):
    try:
        topo, cnn_params, lstm_params, setting = decode_individual(individual)
        batch_size, learn_rate, opt_type, reg_type = decode_hyperparams(setting)

        model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, num_features, sequence_length).to(device)
        model_size = sum(p.numel() for p in model.parameters())
        if model_size > MAX_MODEL_SIZE:
            raise ValueError(f"模型过大: {model_size} 参数")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        criterion = nn.MSELoss()

        if opt_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        elif opt_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
        elif opt_type == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learn_rate)
        else:
            optimizer = optim.Adadelta(model.parameters(), lr=learn_rate)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    raise ValueError("输入数据包含NaN或Inf")

                optimizer.zero_grad()
                outputs = model(inputs)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    raise ValueError("模型输出包含NaN或Inf")

                loss = criterion(outputs, targets)

                if reg_type is not None:
                    l1_penalty = torch.tensor(0., device=device)
                    l2_penalty = torch.tensor(0., device=device)
                    for name, param in model.named_parameters():
                        if 'bias' in name or 'bn' in name:
                            continue
                        if reg_type in ['L1', 'L1L2']:
                            l1_penalty += torch.norm(param, 1)
                        if reg_type in ['L2', 'L1L2']:
                            l2_penalty += torch.norm(param, 2)
                    loss += LAMBDA_REG * (l1_penalty + l2_penalty)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
                optimizer.step()

            model.eval()
            val_loss = 0
            val_samples = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, targets).item() * inputs.size(0)
                    val_loss += batch_loss
                    val_samples += inputs.size(0)
            if val_samples > 0:
                val_loss /= val_samples
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EVAL_PATIENCE:
                break

        model.eval()
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        
        if not all_targets or not all_outputs:
            raise ValueError("没有有效的预测结果")
        
        all_targets = np.array(all_targets)
        all_outputs = np.array(all_outputs)
        all_targets = all_targets * (max_speed - min_speed) + min_speed
        all_outputs = all_outputs * (max_speed - min_speed) + min_speed
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
        if np.isnan(rmse) or np.isinf(rmse):
            raise ValueError(f"无效的RMSE值: {rmse}")
        
        complexity = sum(p.numel() for p in model.parameters())
        return rmse, complexity
    
    except Exception as e:
        print(f"\n{'=' * 50}")
        print(f"评估失败 - 个体ID: {id(individual)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误消息: {e}")
        print("完整堆栈追踪:")
        traceback.print_exc()
        print(f"{'=' * 50}\n")
        return float('inf'), float('inf')


def evaluate_population(population, train_dataset, val_dataset, min_speed, max_speed, num_features, sequence_length,
                        device):
    performance = []
    complexity = []
    for i, ind in enumerate(population):
        print(f"  评估个体 {i + 1}/{len(population)}...")
        rmse, comp = evaluate_individual(ind, train_dataset, val_dataset, min_speed, max_speed, num_features,
                                         sequence_length, device)
        performance.append(rmse)
        complexity.append(comp)
    return np.array(performance), np.array(complexity)