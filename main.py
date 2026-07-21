import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pickle
import time
import warnings

warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False

from src.config import (
    POP_SIZE, MAX_GEN, NUM_RUNS, MUTATION_PROB, FINAL_EPOCHS, FINAL_PATIENCE,
    LAMBDA_REG, SEQUENCE_LENGTH
)
from src.data_loader import load_and_preprocess_data
from src.model import HybridCNNBiLSTM
from src.encoding import decode_individual, decode_hyperparams, initialize_population
from src.nsga2 import (
    fast_non_dominated_sort, crowding_distance, tournament_selection,
    environmental_selection, crossover_population, variable_length_mutation
)
from src.evaluator import evaluate_population
from src.visualizer import generate_comprehensive_report


def main():
    print(f"🚀 即将进行 {NUM_RUNS} 次独立运行（每次使用不同随机种子，确保结果独立可复现）")
    print("每次运行的结果将使用 'run1_'、'run2_' 等前缀单独保存所有文件（pkl、png、csv）\n")

    data_result = load_and_preprocess_data()
    train_dataset = data_result['train_dataset']
    val_dataset = data_result['val_dataset']
    test_dataset = data_result['test_dataset']
    min_speed = data_result['min_speed']
    max_speed = data_result['max_speed']
    num_features = data_result['num_features']
    feature_columns = data_result['feature_columns']

    for run_id in range(NUM_RUNS):
        print(f"\n{'*' * 80}")
        print(f"🚀 开始第 {run_id + 1}/{NUM_RUNS} 次独立运行...")
        print(f"{'*' * 80}\n")

        prefix = f"run{run_id + 1}_"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            print(f'检测到 {torch.cuda.device_count()} 块GPU,将使用多GPU并行')
        print(f'使用设备: {device} (第 {run_id + 1} 次运行)')

        print('\n========== 开始NSGA-II优化自动化深度学习模型 ==========')
        start_time = time.time()

        population = initialize_population(POP_SIZE, train_dataset)
        print(f'初始化完成,种群大小: {len(population)}')

        best_rmse_history = []
        best_complexity_history = []
        all_pareto_fronts = []

        for gen in range(MAX_GEN):
            print(f'\n第 {gen + 1}/{MAX_GEN} 代 (第 {run_id + 1} 次运行)...')
            performance, complexity = evaluate_population(
                population, train_dataset, val_dataset, min_speed, max_speed,
                num_features, SEQUENCE_LENGTH, device
            )
            fronts, rank = fast_non_dominated_sort(performance, complexity)
            distance = crowding_distance(performance, complexity, fronts)
            mating_pool = tournament_selection(population, rank, distance, POP_SIZE)
            offspring = crossover_population(mating_pool)
            mutated_offspring = []
            for child in offspring:
                if np.random.random() < MUTATION_PROB:
                    mutated_child = variable_length_mutation(child)
                    mutated_offspring.append(mutated_child)
                else:
                    mutated_offspring.append(child.copy())
            offspring_perf, offspring_comp = evaluate_population(
                mutated_offspring, train_dataset, val_dataset, min_speed, max_speed,
                num_features, SEQUENCE_LENGTH, device
            )
            combined_pop = population + mutated_offspring
            combined_perf = np.concatenate([performance, offspring_perf])
            combined_comp = np.concatenate([complexity, offspring_comp])
            combined_fronts, combined_rank = fast_non_dominated_sort(combined_perf, combined_comp)
            combined_dist = crowding_distance(combined_perf, combined_comp, combined_fronts)
            population, performance, complexity = environmental_selection(
                combined_pop, combined_perf, combined_comp, combined_rank, combined_dist, POP_SIZE
            )
            pareto_front = {
                'params': [combined_pop[i] for i in combined_fronts[0]],
                'performance': combined_perf[combined_fronts[0]],
                'complexity': combined_comp[combined_fronts[0]],
                'num_solutions': len(combined_fronts[0]),
                'generation': gen + 1
            }
            all_pareto_fronts.append(pareto_front)

            valid_combined_perf = combined_perf[np.isfinite(combined_perf)]
            if len(valid_combined_perf) > 0:
                best_idx_combined = np.argmin(valid_combined_perf)
                best_rmse_history.append(valid_combined_perf[best_idx_combined])
                best_complexity_history.append(combined_comp[best_idx_combined])
            else:
                best_rmse_history.append(float('inf'))
                best_complexity_history.append(float('inf'))

        end_time = time.time()
        print(f'\n第 {run_id + 1} 次优化完成!总耗时: {end_time - start_time:.2f} 秒')

        final_pareto = all_pareto_fronts[-1]
        if final_pareto['num_solutions'] > 0:
            perf_vals = final_pareto['performance']
            comp_vals = final_pareto['complexity']
            valid_mask = np.isfinite(perf_vals) & np.isfinite(comp_vals)
            if valid_mask.any():
                perf_vals = perf_vals[valid_mask]
                comp_vals = comp_vals[valid_mask]
                params = [final_pareto['params'][i] for i in range(len(valid_mask)) if valid_mask[i]]
                normalized_perf = (perf_vals - perf_vals.min()) / (perf_vals.max() - perf_vals.min() + 1e-10)
                normalized_comp = (comp_vals - comp_vals.min()) / (comp_vals.max() - comp_vals.min() + 1e-10)
                trade_off_scores = np.sqrt(normalized_perf ** 2 + normalized_comp ** 2)
                best_idx = np.argmin(trade_off_scores)
                best_individual = params[best_idx]
            else:
                raise ValueError("最后一代没有有效解")
        else:
            all_perf = np.concatenate([pf['performance'] for pf in all_pareto_fronts])
            all_params = [p for pf in all_pareto_fronts for p in pf['params']]
            valid_mask = np.isfinite(all_perf)
            if valid_mask.any():
                best_idx = np.argmin(all_perf[valid_mask])
                best_individual = np.array(all_params)[valid_mask][best_idx]
            else:
                raise ValueError("所有个体评估均失败!")

        topo, cnn_params, lstm_params, setting = decode_individual(best_individual)
        batch_size, learn_rate, opt_type, reg_type = decode_hyperparams(setting)

        print(f'\n第 {run_id + 1} 次运行最终优化的超参数数值:')
        print(f' batch_size: {batch_size}')
        print(f' learning_rate: {learn_rate:.6f}')
        print(f' optimizer: {opt_type}')
        print(f' regularizer: {reg_type}')

        print('\n训练最终模型...')
        model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, num_features, SEQUENCE_LENGTH).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        criterion = nn.MSELoss()

        if opt_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        elif opt_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
        elif opt_type == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learn_rate)
        else:
            optimizer = optim.Adadelta(model.parameters(), lr=learn_rate)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(FINAL_EPOCHS):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
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
                optimizer.step()
                train_loss += loss.item()

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
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            if (epoch + 1) % 5 == 0:
                print(f' Epoch {epoch + 1}/{FINAL_EPOCHS}, 训练损失: {train_loss / len(train_loader):.6f}, 验证损失: {val_loss:.6f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= FINAL_PATIENCE:
                print(f'早停于 epoch {epoch + 1}')
                break

        print('\n正在绘制训练损失曲线...')
        plt.figure(figsize=(12, 8))
        epochs_range = range(1, len(train_losses) + 1)
        plt.plot(epochs_range, train_losses, 'b-', linewidth=2, label='训练损失', alpha=0.8)
        plt.plot(epochs_range, val_losses, 'r--', linewidth=2, label='验证损失', alpha=0.8)
        plt.xlabel('训练轮次 (Epoch)', fontsize=14)
        plt.ylabel('损失值 (MSE)', fontsize=14)
        plt.title('模型训练过程损失曲线', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{prefix}training_loss_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

        model.eval()
        all_test_targets = []
        all_test_outputs = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                all_test_targets.extend(targets.cpu().numpy())
                all_test_outputs.extend(outputs.cpu().numpy())
        all_test_targets = np.array(all_test_targets) * (max_speed - min_speed) + min_speed
        all_test_outputs = np.array(all_test_outputs) * (max_speed - min_speed) + min_speed
        all_test_targets = all_test_targets.flatten()
        all_test_outputs = all_test_outputs.flatten()

        mae_test = mean_absolute_error(all_test_targets, all_test_outputs)
        rmse_test = np.sqrt(mean_squared_error(all_test_targets, all_test_outputs))
        mape_test = np.mean(np.abs((all_test_targets - all_test_outputs) / all_test_targets)) * 100
        r2_test = r2_score(all_test_targets, all_test_outputs)
        r_test = pearsonr(all_test_targets, all_test_outputs)[0]

        print(f'\n第 {run_id + 1} 次运行测试集最终性能:')
        print(f' MAE: {mae_test:.4f} m/s')
        print(f' RMSE: {rmse_test:.4f} m/s')
        print(f' MAPE: {mape_test:.2f}%')
        print(f' R²: {r2_test:.4f}')
        print(f' 相关系数: {r_test:.4f}')

        test_performance = {'mae': mae_test, 'rmse': rmse_test, 'mape': mape_test}

        print('\n正在绘制 Pareto 前沿演化图...')
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_pareto_fronts)))
        for idx, pf in enumerate(all_pareto_fronts):
            if pf['num_solutions'] > 0:
                perf = pf['performance']
                comp = pf['complexity']
                valid_mask = np.isfinite(perf) & np.isfinite(comp)
                if np.any(valid_mask):
                    plt.scatter(comp[valid_mask], perf[valid_mask], c=[colors[idx]], s=60, alpha=0.6,
                                label=f'第 {pf["generation"]} 代', edgecolors='k', linewidth=0.5)
        plt.xlabel('模型复杂度 (参数数量)', fontsize=14)
        plt.ylabel('验证集 RMSE (m/s)', fontsize=14)
        plt.title('NSGA-II Pareto 前沿演化过程', fontsize=16)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{prefix}pareto_evolution_continuous_lr.png', dpi=300, bbox_inches='tight')
        plt.show()

        print('\n正在绘制关键代数Pareto前沿对比图（带连接线）...')
        target_generations = [1, 5, 10, 20, 25, 30]
        plt.figure(figsize=(14, 10))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        plotted_any = False
        for idx, target_gen in enumerate(target_generations):
            pf = None
            for front in all_pareto_fronts:
                if front['generation'] == target_gen:
                    pf = front
                    break
            if pf is None or pf['num_solutions'] == 0:
                continue
            perf = pf['performance']
            comp = pf['complexity']
            valid_mask = np.isfinite(perf) & np.isfinite(comp)
            if not np.any(valid_mask):
                continue
            perf = perf[valid_mask]
            comp = comp[valid_mask]
            sort_idx = np.argsort(comp)
            comp_sorted = comp[sort_idx]
            perf_sorted = perf[sort_idx]
            plt.plot(comp_sorted, perf_sorted, color=colors[idx], linewidth=2, alpha=0.7,
                     label=f'第 {target_gen} 代 (n={len(perf)})')
            plt.scatter(comp, perf, c=colors[idx], marker=markers[idx], s=80, alpha=0.9,
                        edgecolors='black', linewidth=0.5)
            plotted_any = True
        if plotted_any:
            plt.xlabel('模型复杂度 (参数数量)', fontsize=14)
            plt.ylabel('验证集 RMSE (m/s)', fontsize=14)
            plt.title('NSGA-II Pareto前沿进化过程对比（带连接线）', fontsize=16, fontweight='bold')
            plt.legend(fontsize=11, loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.figtext(0.5, 0.02,
                        '注：第1代为初始种群，第30代为最终进化结果\n每条实线连接该代的所有Pareto最优解，展示前沿面形状',
                        ha='center', fontsize=10, style='italic')
            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            plt.savefig(f'{prefix}pareto_evolution_comparison_connected.png', dpi=300, bbox_inches='tight')
        plt.show()

        print('\n正在绘制最终代Pareto前沿详细图（第30代）...')
        final_pf = None
        for pf in all_pareto_fronts:
            if pf['generation'] == 30:
                final_pf = pf
                break
        if final_pf and final_pf['num_solutions'] > 0:
            perf_final = final_pf['performance']
            comp_final = final_pf['complexity']
            valid_final = np.isfinite(perf_final) & np.isfinite(comp_final)
            if np.any(valid_final):
                perf_final = perf_final[valid_final]
                comp_final = comp_final[valid_final]
                sort_idx = np.argsort(comp_final)
                comp_sorted = comp_final[sort_idx]
                perf_sorted = perf_final[sort_idx]
                plt.figure(figsize=(12, 8))
                plt.plot(comp_sorted, perf_sorted, color='darkred', linewidth=3, alpha=0.8,
                         marker='o', markersize=10, markerfacecolor='red',
                         markeredgecolor='black', markeredgewidth=1.5)
                plt.xlabel('模型复杂度 (参数数量)', fontsize=14)
                plt.ylabel('验证集 RMSE (m/s)', fontsize=14)
                plt.title('最终Pareto前沿（第30代）', fontsize=16, fontweight='bold')
                plt.grid(True, alpha=0.3)
                for i, (comp_val, perf_val) in enumerate(zip(comp_sorted, perf_sorted)):
                    plt.annotate(f'({comp_val:.0f}, {perf_val:.3f})', xy=(comp_val, perf_val),
                                 xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
                plt.tight_layout()
                plt.savefig(f'{prefix}pareto_front_final_gen30.png', dpi=300, bbox_inches='tight')
                plt.show()

        with open(f'{prefix}modeo_cnn_optimization_continuous_lr.pkl', 'wb') as f:
            pickle.dump({
                'best_individual': best_individual,
                'best_rmse_history': best_rmse_history,
                'pareto_fronts': all_pareto_fronts,
                'test_performance': test_performance,
                'training_losses': train_losses,
                'validation_losses': val_losses,
                'model_params': {
                    'topo': topo,
                    'cnn_params': cnn_params,
                    'lstm_params': lstm_params,
                    'setting': setting,
                    'batch_size': batch_size,
                    'learn_rate': learn_rate,
                    'opt_type': opt_type
                },
                'run_id': run_id + 1,
            }, f)

        report_metrics = generate_comprehensive_report(
            model, test_loader, val_loader, device, min_speed, max_speed,
            all_pareto_fronts, feature_columns, topo, cnn_params, lstm_params, setting,
            test_performance, r_test, save_prefix=prefix
        )

        print('\n========== 代码运行结束后保存文件 ==========')
        test_results_df = pd.DataFrame({
            '真实风速 (m/s)': all_test_targets,
            '预测风速 (m/s)': all_test_outputs
        })
        test_results_df.to_csv(f'{prefix}test_set_wind_speed_predictions.csv', index=False, encoding='utf-8-sig')

        for pf in all_pareto_fronts:
            gen = pf['generation']
            perf = pf['performance']
            comp = pf['complexity']
            valid_mask = np.isfinite(perf) & np.isfinite(comp)
            if np.any(valid_mask):
                pareto_df = pd.DataFrame({
                    '模型复杂度 (参数数量)': comp[valid_mask],
                    '验证集RMSE (m/s)': perf[valid_mask]
                })
                pareto_df = pareto_df.sort_values(by='模型复杂度 (参数数量)')
                filename = f'{prefix}pareto_front_generation_{gen}.csv'
                pareto_df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"\n{'#' * 80}")
    print(f"🎉 所有 {NUM_RUNS} 次独立运行已完成！")
    print(f"{'#' * 80}")


if __name__ == '__main__':
    main()