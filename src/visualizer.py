import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import torch
import torch.nn as nn


def generate_comprehensive_report(model, test_loader, val_loader, device, min_speed, max_speed, all_pareto_fronts,
                                  feature_columns, topo, cnn_params, lstm_params, setting, test_performance, r_test,
                                  save_prefix=''):
    print('\n========== 生成综合预测报告 ==========')
    model.eval()
    
    all_val_targets = []
    all_val_outputs = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_val_targets.extend(targets.cpu().numpy())
            all_val_outputs.extend(outputs.cpu().numpy())
    
    all_val_targets = np.array(all_val_targets).flatten()
    all_val_outputs = np.array(all_val_outputs).flatten()
    all_val_targets = all_val_targets * (max_speed - min_speed) + min_speed
    all_val_outputs = all_val_outputs * (max_speed - min_speed) + min_speed

    all_test_targets = []
    all_test_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_test_targets.extend(targets.cpu().numpy())
            all_test_outputs.extend(outputs.cpu().numpy())
    
    all_test_targets = np.array(all_test_targets).flatten()
    all_test_outputs = np.array(all_test_outputs).flatten()
    all_test_targets = all_test_targets * (max_speed - min_speed) + min_speed
    all_test_outputs = all_test_outputs * (max_speed - min_speed) + min_speed

    val_residuals = all_val_targets - all_val_outputs
    test_residuals = all_test_targets - all_test_outputs

    mae_val = mean_absolute_error(all_val_targets, all_val_outputs)
    rmse_val = np.sqrt(mean_squared_error(all_val_targets, all_val_outputs))
    mape_val = np.mean(np.abs(val_residuals / all_val_targets)) * 100
    r_val = pearsonr(all_val_targets, all_val_outputs)[0]

    mae_test = test_performance['mae']
    rmse_test = test_performance['rmse']
    mape_test = test_performance['mape']
    r_test = pearsonr(all_test_targets, all_test_outputs)[0]

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('CNN-BiLSTM 模型预测效果综合报告', fontsize=20, fontweight='bold', y=0.98)
    
    ax1 = plt.subplot(2, 3, 1)
    sample_idx = np.arange(0, min(500, len(all_val_targets)))
    ax1.plot(sample_idx, all_val_targets[sample_idx], 'b-', linewidth=1.5, label='真实值', alpha=0.8)
    ax1.plot(sample_idx, all_val_outputs[sample_idx], 'r--', linewidth=1.5, label='预测值', alpha=0.8)
    ax1.set_xlabel('时间步', fontsize=12)
    ax1.set_ylabel('风速 (m/s)', fontsize=12)
    ax1.set_title(f'验证集预测效果\n(RMSE={rmse_val:.3f}, MAE={mae_val:.3f})', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    sample_idx = np.arange(0, min(500, len(all_test_targets)))
    ax2.plot(sample_idx, all_test_targets[sample_idx], 'b-', linewidth=1.5, label='真实值', alpha=0.8)
    ax2.plot(sample_idx, all_test_outputs[sample_idx], 'r--', linewidth=1.5, label='预测值', alpha=0.8)
    ax2.set_xlabel('时间步', fontsize=12)
    ax2.set_ylabel('风速 (m/s)', fontsize=12)
    ax2.set_title(f'测试集预测效果\n(RMSE={rmse_test:.3f}, MAE={mae_test:.3f})', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(test_residuals, bins=50, color='darkcyan', alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.mean(test_residuals), color='red', linestyle='--', linewidth=2,
                label=f'均值: {np.mean(test_residuals):.3f}')
    ax3.set_xlabel('残差 (真实值 - 预测值)', fontsize=12)
    ax3.set_ylabel('频数', fontsize=12)
    ax3.set_title('测试集残差分布', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_residuals, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
    ax4.set_xlabel('预测误差 (m/s)', fontsize=12)
    ax4.set_ylabel('概率密度', fontsize=12)
    ax4.set_title('测试集误差概率密度', fontsize=14)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    final_pareto = all_pareto_fronts[-1]
    if final_pareto['num_solutions'] > 0:
        perf_vals = final_pareto['performance']
        comp_vals = final_pareto['complexity']
        valid_mask = np.isfinite(perf_vals) & np.isfinite(comp_vals)
        if np.any(valid_mask):
            ax5.scatter(comp_vals[valid_mask], perf_vals[valid_mask], c='darkgreen', s=100, edgecolors='black',
                        linewidth=1, alpha=0.8)
            ax5.set_xlabel('模型复杂度 (参数数量)', fontsize=12)
            ax5.set_ylabel('验证集 RMSE (m/s)', fontsize=12)
            ax5.set_title('最后一代Pareto前沿', fontsize=14)
            ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    metrics = ['RMSE', 'MAE', 'MAPE', 'R']
    test_metrics = [rmse_test, mae_test, mape_test, r_test]
    x = np.arange(len(metrics))
    width = 0.35
    bars2 = ax6.bar(x + width / 2, test_metrics, width, label='测试集', color='darkcyan', alpha=0.8)
    ax6.set_xlabel('评估指标', fontsize=12)
    ax6.set_ylabel('指标值', fontsize=12)
    ax6.set_title('性能指标对比', fontsize=14)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)

    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    add_value_labels(ax6, bars2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{save_prefix}comprehensive_cnn_bilstm_report.png', dpi=300, bbox_inches='tight')
    plt.show()

    if isinstance(model, nn.DataParallel):
        feature_weights = model.module.feature_weights.detach().cpu().numpy()
    else:
        feature_weights = model.feature_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    weights = feature_weights.flatten()
    plt.bar(range(len(feature_columns)), weights, color=[0.2, 0.5, 0.8])
    plt.xticks(range(len(feature_columns)), feature_columns, rotation=15)
    plt.xlabel('输入特征')
    plt.ylabel('学习到的权重（越大越重要）')
    plt.title('HybridCNNBiLSTM 中各输入特征的权重')
    plt.grid(True, axis='y')
    for i, w in enumerate(weights):
        plt.text(i, w + 0.02, f'{w:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}feature_weights.png', dpi=300)
    plt.show()

    print('\n========== 详细性能报告 ==========')
    print(f'验证集性能:')
    print(f' MAE: {mae_val:.4f} m/s')
    print(f' RMSE: {rmse_val:.4f} m/s')
    print(f' MAPE: {mape_val:.2f}%')
    print(f' R (皮尔逊相关系数): {r_val:.4f}')
    print(f'\n测试集性能:')
    print(f' MAE: {mae_test:.4f} m/s')
    print(f' RMSE: {rmse_test:.4f} m/s')
    print(f' MAPE: {mape_test:.2f}%')
    print(f' R (皮尔逊相关系数): {r_test:.4f}')
    print(f'\n特征重要性排名:')
    feature_importance = sorted(zip(feature_columns, weights), key=lambda x: x[1], reverse=True)
    for i, (feature, weight) in enumerate(feature_importance, 1):
        print(f' {i}. {feature}: {weight:.4f}')
    print(f'\n综合报告已生成并保存！前缀: {save_prefix}')
    
    return {
        'val_metrics': {'mae': mae_val, 'rmse': rmse_val, 'mape': mape_val, 'r': r_val},
        'test_metrics': {'mae': mae_test, 'rmse': rmse_test, 'mape': mape_test, 'r': r_test}
    }