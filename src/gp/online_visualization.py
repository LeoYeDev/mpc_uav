import matplotlib.pyplot as plt
import numpy as np
import torch
import gpytorch
import os
from src.visualization.style import set_publication_style
from config.configuration_parameters import DirectoryConfig
from matplotlib.ticker import StrMethodFormatter # <-- 新增导入

# 注意: 如果 gp_common 模块无法找到，请确保正确设置了 PYTHONPATH
try:
    from src.gp.utils import world_to_body_velocity_mapping
except ImportError:
    print("警告：无法从 src.gp.utils 导入，将使用本地的虚拟函数。")


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
    set_publication_style(base_size=9)  # 设置专业的出版物风格

    num_dims = online_gp_manager.num_dimensions
    fig, axes = plt.subplots(num_dims, 1, squeeze=False, figsize=(3.5, 1.5 * num_dims), constrained_layout=True)
    #fig.suptitle(f"Online Gaussian Process State Snapshot: {snapshot_info_str}", fontsize=20, weight='bold')

    axis_map = {0: 'Vx', 1: 'Vy', 2: 'Vz'}
    colors = {
        'fit_mean': '#e74c3c',
        'fit_ci': "#e74c3c",
        'train_bubble_face': "#a9bbf1",
        'train_bubble_edge': "#3498db", # 更深的蓝色边缘
        'predict_bubble_face': "#ffffff", # 更鲜艳的红色
        'predict_bubble_edge': '#d62728', # 更深的红色边缘
        'predict_ci_bar': "#d18c8c",     # 更明显的粉红色误差棒
        'buffer_history': 'lightgrey' # 新增：为历史数据点定义颜色
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
            ax.grid(True)
            continue
        
        # ================================================================
        # --- 新增代码部分：绘制缓冲区内的所有历史数据点 ---
        # ================================================================
        # all_buffer_points = []
        # all_buffer_points = list(gp.full_history_buffer)
        
        # if all_buffer_points:
        #     # 提取x和y坐标
        #     buffer_x = np.array([p[0] for p in all_buffer_points])
        #     buffer_y = np.array([p[1] for p in all_buffer_points])
        #     # 使用小的、半透明的灰色点来绘制历史数据
        #     ax.scatter(buffer_x, buffer_y, s=25, 
        #                facecolors=colors['buffer_history'], alpha=0.5,
        #                label='Buffer History', zorder=1)
        # ================================================================
        # --- 新增代码结束 ---
        # ================================================================

        # a) 绘制当前缓冲区中的“真实”训练数据点 (蓝色气泡)
        train_x_denorm, train_y_denorm = None, None
        training_data_raw = gp.buffer.get_training_set()
        if training_data_raw:
            train_x_denorm = np.array([p[0] for p in training_data_raw])
            train_y_denorm = np.array([p[1] for p in training_data_raw])
            ax.scatter(train_x_denorm, train_y_denorm, s=20, 
                       facecolors=colors['train_bubble_face'], alpha=0.8,
                       edgecolors=colors['train_bubble_edge'], linewidth=1.0, # 加粗边缘
                       label='Data', zorder=4)

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

            ax.plot(x_dense_denorm, fit_mean_dim_i, color=colors['fit_mean'], lw=2, label='GP Fit', zorder=3) # 增加线宽
            
            ax.fill_between(x_dense_denorm, fit_mean_dim_i - 1.96 * fit_std_dim_i, fit_mean_dim_i + 1.96 * fit_std_dim_i,
                            color=colors['fit_ci'], alpha=0.2, label='95% CI',zorder=2) # 增加透明度

        # c) 绘制MPC规划点的预测结果 (红色气泡)
        pred_inputs_dim_i = future_velocities_np[:, i]
        mean_pred_dim_i = mpc_plan_means[:, i]
        std_pred_dim_i = np.sqrt(np.maximum(mpc_plan_vars[:, i], 1e-9))
        
        # 绘制更醒目的误差棒
        # ax.errorbar(pred_inputs_dim_i, mean_pred_dim_i, yerr=1.96 * std_pred_dim_i,
        #             fmt='none', ecolor=colors['predict_ci_bar'], elinewidth=5, # 加粗误差棒
        #             capsize=0, zorder=11, alpha=0.8, label='_nolegend_') # 增加透明度, 且不在图例中显示
        # 绘制更醒目的红色气泡
        ax.scatter(pred_inputs_dim_i, mean_pred_dim_i, s=30, # 增大尺寸
                   facecolors=colors['predict_bubble_face'], alpha=0.8, # 增加不透明度
                   edgecolors=colors['predict_bubble_edge'], linewidth=1.0, # 加粗边缘
                   label='Pred', zorder=5)

        # d) 图表元素精细化调整
        x_sub = axis_name.lower().replace('v', '') # 从 'Vx' 中提取 'x'
        x_label = fr'$v_{{{x_sub}}}$ [m/s]'
        y_label = fr'$\Delta a_{{{x_sub}}}$ [m/s²]'
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        ax.grid(True)
        ax.axhline(0, color='black', lw=1.0, linestyle='--', alpha=0.7)

        # --- 新增：强制Y轴刻度标签显示为一位小数的浮点数 ---
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:1.1f}'))

    # 图例设置
    fig.tight_layout() 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',      # 定位在顶部中央
               bbox_to_anchor=(0.5, 0.92), # 精确控制位置
               ncol=len(handles),       # 实现水平布局
               frameon=True,
               handlelength=1.2,            # 图例线的长度（可调）
               columnspacing=1.2,           # 列间距
               borderpad=0.3                # 边框内边距
               )      
    # 调整子图间距（垂直间距hspace是关键）
    plt.subplots_adjust(
        hspace=0.5,  # 增加子图间距
        top=0.87     # 预留顶部空间给图例
    )
    
    fig_path = os.path.join(DirectoryConfig.FIGURES_DIR, 'online_gp_snapshot')
    plt.savefig(fig_path + '.pdf', bbox_inches="tight")
    plt.savefig(fig_path + '.svg', bbox_inches="tight")
    plt.show(block=False)

    # --- 3. 任务二：绘制独立的、仅含X轴的精美快照 ---
    print("\n--- 正在生成仅X轴的GP快照... ---")
    set_publication_style(base_size=9)
    fig_x, ax_x = plt.subplots(1, 1, figsize=(3, 2), dpi=150)

    # 仅针对 X 轴 (索引 i=0) 进行绘图
    i = 0  # X轴索引 (0=X, 1=Y, 2=Z)
    gp = online_gp_manager.gps[i]

    if not gp.is_trained_once:
        ax_x.text(0.5, 0.5, 'Waiting for initial data for X-axis...', 
                  ha='center', va='center', transform=ax_x.transAxes, fontsize=14, color='grey')
        ax_x.grid(True)
    else:
        # a) 绘制训练数据点
        training_data_raw = gp.buffer.get_training_set()
        if training_data_raw:
            train_x_denorm = np.array([p[0] for p in training_data_raw])
            train_y_denorm = np.array([p[1] for p in training_data_raw])
            ax_x.scatter(train_x_denorm, train_y_denorm, s=30, 
                       facecolors=colors['train_bubble_face'], alpha=0.8,
                       edgecolors=colors['train_bubble_edge'], linewidth=0.7,
                       label='Data', zorder=4)

        # b) 绘制GP拟合曲线
        if 'train_x_denorm' in locals() and train_x_denorm.size > 0:
            x_min_plot, x_max_plot = train_x_denorm.min(), train_x_denorm.max()
            range_ext = (x_max_plot - x_min_plot) * 0.15
            if abs(range_ext) < 1e-4: range_ext = 0.5
            x_dense_denorm = np.linspace(x_min_plot - range_ext, x_max_plot + range_ext, 200)
            
            v_std_ema = np.sqrt(gp.v_var_ema + gp.epsilon)
            x_dense_norm = (x_dense_denorm - gp.v_mean_ema) / v_std_ema
            model_dtype = next(gp.model.parameters()).dtype
            x_dense_torch = torch.tensor(x_dense_norm, device=gp.device, dtype=model_dtype).view(-1, 1)

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = gp.likelihood(gp.model(x_dense_torch))
                mean_norm = preds.mean.cpu().numpy()
                var_norm = preds.variance.cpu().numpy()
        
            r_std_ema = np.sqrt(gp.r_var_ema + gp.epsilon)
            fit_mean_dim_i = mean_norm * r_std_ema + gp.r_mean_ema
            fit_std_dim_i = np.sqrt(np.maximum(var_norm * (gp.r_var_ema + gp.epsilon), 1e-9))

            ax_x.plot(x_dense_denorm, fit_mean_dim_i, color=colors['fit_mean'], lw=1.0, label='GP Fit', zorder=3)
            ax_x.fill_between(x_dense_denorm, fit_mean_dim_i - 1.96 * fit_std_dim_i, fit_mean_dim_i + 1.96 * fit_std_dim_i,
                            color=colors['fit_ci'], alpha=0.2, label='95% CI', zorder=2)

        # c) 绘制MPC规划点的预测结果
        pred_inputs_dim_i = future_velocities_np[:, i]
        mean_pred_dim_i = mpc_plan_means[:, i]
        ax_x.scatter(pred_inputs_dim_i, mean_pred_dim_i, s=40,
                   facecolors=colors['predict_bubble_face'], alpha=0.9,
                   edgecolors=colors['predict_bubble_edge'], linewidth=1.2,
                   label='Pred', zorder=5)

        # d) 图表元素精细化调整
        x_label = r'$v_{x}$ [m/s]'
        y_label = r'$\Delta a_{x}$ [m/s²]'
        ax_x.set_xlabel(x_label, fontsize=12)
        ax_x.set_ylabel(y_label, fontsize=12)
        ax_x.grid(True)
        ax_x.axhline(0, color='black', lw=1.2, linestyle='--', alpha=0.7)
        ax_x.yaxis.set_major_formatter(StrMethodFormatter('{x:1.1f}'))
        ax_x.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(DirectoryConfig.FIGURES_DIR, 'online_gp_snapshot_x_only')
    plt.savefig(fig_path + '.pdf', bbox_inches="tight")
    plt.savefig(fig_path + '.svg', bbox_inches="tight")
    plt.show(block=False)
    print("--- 仅X轴图已生成。---")