import matplotlib.pyplot as plt
import numpy as np
import torch
import gpytorch

# 假设项目的根目录 'src' 已经被添加到了 PYTHONPATH
# 注意: 如果 gp_common 模块无法找到，请确保正确设置了 PYTHONPATH
try:
    from src.model_fitting.gp_common import world_to_body_velocity_mapping
except ImportError:
    # 提供一个备用方案，以防在不同环境下运行此脚本
    print("警告：无法从 src.model_fitting.gp_common 导入，将使用本地的虚拟函数。")
    def world_to_body_velocity_mapping(states):
        # 这是一个虚拟实现，仅用于代码不报错。
        # 在实际项目中，请确保 PYTHONPATH 设置正确。
        return states

def visualize_gp_snapshot(online_gp_manager, mpc_planned_states, snapshot_info_str, **kwargs):
    """
    创建一个经过深度美化的在线GP模型快照可视化，专为学术论文优化，采用精美的气泡样式。

    该函数会为每个输出维度生成一个子图，以极高的清晰度展示：
    1.  当前缓冲区中的训练数据点 (Training Data) - 蓝色气泡样式。
    2.  GP对训练数据所在区域的拟合曲线及95%置信区间 (GP Fit & 95% CI)。
    3.  GP对未来MPC规划速度的预测值及置信区间 (MPC Plan Prediction) - 红色气泡样式。

    Args:
        online_gp_manager (IncrementalGPManager): 我们最终版的、正在运行的在线GP管理器实例。
        mpc_planned_states (np.ndarray): MPC规划的未来状态轨迹(N, 13)，通常在世界坐标系下。
        snapshot_info_str (str): 显示在图表顶部的标题信息，如当前仿真时间。
        **kwargs: 预留的其他参数。
    """
    # --- 1. 数据准备：与原逻辑保持一致 ---
    try:
        planned_states_body = world_to_body_velocity_mapping(mpc_planned_states.copy())
        future_velocities_np = planned_states_body[:, 7:10]
    except Exception as e:
        print(f"⚠️ [可视化错误] 坐标变换失败: {e}")
        return
        
    # 在可视化代码中直接调用管理器的预测方法，获取MPC规划点的预测结果
    mpc_plan_means, mpc_plan_vars = online_gp_manager.predict(future_velocities_np)

    # --- 2. 图表风格设定：注入专业学术感 ---
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except IOError:
        plt.style.use('default')

    try:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 14, # 稍微增大基础字号，提高可读性
        })
    except Exception as e:
        print(f"⚠️ [可视化] 设置字体失败: {e}。将使用默认字体。")

    num_dims = online_gp_manager.num_dimensions
    fig, axes = plt.subplots(num_dims, 1, figsize=(12, 5 * num_dims), squeeze=False, dpi=150)
    #fig.suptitle(f"Online Gaussian Process State Snapshot: {snapshot_info_str}", fontsize=20, weight='bold')

    axis_map = {0: 'Vx', 1: 'Vy', 2: 'Vz'}
    colors = {
        'fit_mean': '#0072BD',
        'fit_ci': '#A2C8EC',
        'train_bubble_face': '#2E86C1',
        'train_bubble_edge': '#1B4F72',
        'predict_bubble_face': '#E74C3C', # 更鲜艳的红色
        'predict_bubble_edge': '#943126', # 更深的红色边缘
        'predict_ci_bar': "#DE5C4E",     # 更明显的粉红色误差棒
    }

    # --- 3. 逐维度精细绘图 ---
    for i in range(num_dims):
        ax = axes[i, 0]
        gp = online_gp_manager.gps[i]
        axis_name = axis_map.get(i, f"Dim {i}")

        if not gp.is_trained_once:
            # 移除子图标题，仅在Y轴标签上注明维度
            ax.set_ylabel(f'Accel. Residual on {axis_name}\n(Not Trained Yet)', fontsize=14, style='italic')
            ax.text(0.5, 0.5, 'Waiting for initial data...', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='grey')
            ax.grid(True, linestyle='--', alpha=0.5)
            continue

        # a) 绘制当前缓冲区中的“真实”训练数据点 (蓝色气泡)
        train_x_denorm, train_y_denorm = None, None
        training_data_raw = gp.buffer.get_training_set()
        if training_data_raw:
            train_x_denorm = np.array([p[0] for p in training_data_raw])
            train_y_denorm = np.array([p[1] for p in training_data_raw])
            ax.scatter(train_x_denorm, train_y_denorm, s=60, 
                       facecolors=colors['train_bubble_face'], alpha=0.6,
                       edgecolors=colors['train_bubble_edge'], linewidth=1.5,
                       label='Training Data', zorder=10)

        # b) 绘制GP在训练数据范围内的拟合曲线
        if train_x_denorm is not None and train_x_denorm.size > 0:
            x_min_plot, x_max_plot = train_x_denorm.min(), train_x_denorm.max()
            range_ext = (x_max_plot - x_min_plot) * 0.15
            if abs(range_ext) < 1e-4: range_ext = 0.5
            x_dense_denorm = np.linspace(x_min_plot - range_ext, x_max_plot + range_ext, 200)
            
            # 为每个独立GP执行完整的预测流程
            v_std_ema = np.sqrt(gp.v_var_ema + gp.epsilon)
            x_dense_norm = (x_dense_denorm - gp.v_mean_ema) / v_std_ema
            model_dtype = next(gp.model.parameters()).dtype
            x_dense_torch = torch.tensor(x_dense_norm, device=gp.device, dtype=model_dtype).view(-1, 1)

            gp.model.eval()
            gp.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = gp.likelihood(gp.model(x_dense_torch))
                mean_norm = preds.mean.cpu().numpy()
                var_norm = preds.variance.cpu().numpy()
        
            r_std_ema = np.sqrt(gp.r_var_ema + gp.epsilon)
            fit_mean_dim_i = mean_norm * r_std_ema + gp.r_mean_ema
            fit_var_dim_i = var_norm * (gp.r_var_ema + gp.epsilon)
            fit_std_dim_i = np.sqrt(np.maximum(fit_var_dim_i, 1e-9))

            ax.plot(x_dense_denorm, fit_mean_dim_i, color=colors['fit_mean'], lw=3, label='GP Fit Mean', zorder=5) # 增加线宽
            
            ax.fill_between(x_dense_denorm, fit_mean_dim_i - 1.96 * fit_std_dim_i, fit_mean_dim_i + 1.96 * fit_std_dim_i,
                            color=colors['fit_ci'], alpha=0.5, label='95% Confidence Interval') # 增加透明度

        # c) 绘制MPC规划点的预测结果 (红色气泡)
        pred_inputs_dim_i = future_velocities_np[:, i]
        mean_pred_dim_i = mpc_plan_means[:, i]
        std_pred_dim_i = np.sqrt(np.maximum(mpc_plan_vars[:, i], 1e-9))
        
        # 绘制更醒目的误差棒
        ax.errorbar(pred_inputs_dim_i, mean_pred_dim_i, yerr=1.96 * std_pred_dim_i,
                    fmt='none', ecolor=colors['predict_ci_bar'], elinewidth=3, # 加粗误差棒
                    capsize=0, zorder=11, alpha=0.8, label='_nolegend_') # 增加透明度, 且不在图例中显示
        # 绘制更醒目的红色气泡
        ax.scatter(pred_inputs_dim_i, mean_pred_dim_i, s=80, # 增大尺寸
                   facecolors=colors['predict_bubble_face'], alpha=0.85, # 增加不透明度
                   edgecolors=colors['predict_bubble_edge'], linewidth=2.0, # 加粗边缘
                   label='MPC Plan Prediction', zorder=12)

        # d) 图表元素精细化调整
        ax.set_xlabel(f"{axis_name}-axis (m/s)", fontsize=14)
        ax.set_ylabel(f"{axis_name} (m/s²)", fontsize=14)
        
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.axhline(0, color='black', lw=1.0, linestyle='--', alpha=0.7)

     # --- 4. 创建一个共享的、水平的、位于顶部的图例 (如图所示) ---
    # 从最后一个子图获取图例句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 使用 fig.legend() 在整个图表的顶部创建图例
    fig.legend(handles, labels,
               loc='upper center',      # 定位在顶部中央
               bbox_to_anchor=(0.5, 0.98), # 精确控制位置
               ncol=len(handles),       # 实现水平布局
               frameon=True,            # *** 核心修改: 设置为True来显示图例边框 ***
               edgecolor='black',       # 明确边框颜色为黑色
               fontsize=14)             # 设置字体大小


    # 调整子图布局，为顶部的标题和图例留出空间
    plt.subplots_adjust(hspace=0.4, top=0.90, bottom=0.08)
    
    # 调整子图布局，增加垂直间距
    plt.subplots_adjust(hspace=0.4, top=0.92, bottom=0.08)
    plt.show(block=False)
    plt.pause(0.1)
