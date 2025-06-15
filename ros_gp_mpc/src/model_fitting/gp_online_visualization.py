# 文件: src/model_fitting/gp_online_visualization.py
import matplotlib.pyplot as plt
import numpy as np
import torch

# 假设项目的根目录 'src' 已经被添加到了 PYTHONPATH
from src.model_fitting.gp_common import world_to_body_velocity_mapping

def visualize_gp_snapshot(online_gp_manager, mpc_planned_states, snapshot_info_str, **kwargs):
    """
    创建一个经过美化的在线GP模型快照可视化。

    该函数会为每个输出维度生成一个子图，展示：
    1.  当前缓冲区中的训练数据点 (Current Training Points)。
    2.  GP对训练数据所在区域的拟合曲线及置信区间 (GP Fit on Training Domain)。
    3.  GP对未来MPC规划速度的预测值及置信区间 (GP Prediction on MPC Plan)。

    Args:
        online_gp_manager (IncrementalGPManager): 我们最终版的、正在运行的在线GP管理器实例。
        mpc_planned_states (np.ndarray): MPC规划的未来状态轨迹(N, 13)，通常在世界坐标系下。
        snapshot_info_str (str): 显示在图表顶部的标题信息，如当前仿真时间。
        **kwargs: 预留的其他参数。
    """
    # --- 1. 准备MPC规划点的预测数据 ---
    try:
        # 将世界坐标系下的规划状态，转换为机体坐标系下的速度
        planned_states_body = world_to_body_velocity_mapping(mpc_planned_states.copy())
        future_velocities_np = planned_states_body[:, 7:10]
    except Exception as e:
        print(f"⚠️ [可视化错误] 坐标变换失败: {e}")
        return

    # 使用GP管理器的公共接口，对MPC规划的速度点进行预测
    mpc_plan_means, mpc_plan_vars = online_gp_manager.predict(future_velocities_np)

    # --- 2. 创建图表并设置样式 ---
    num_dims = online_gp_manager.num_dimensions
    # 使用兼容性更好的样式名
    try:
        plt.style.use('seaborn-whitegrid')
    except IOError:
        print("⚠️ [可视化] 'seaborn-whitegrid' 样式未找到，使用默认样式。")
        plt.style.use('default')

    fig, axes = plt.subplots(num_dims, 1, figsize=(16, 8 * num_dims), squeeze=False, dpi=120)
    fig.suptitle(f"Online GP Learning Snapshot: {snapshot_info_str}", fontsize=20, weight='bold')
    
    # 统一定义颜色和标签
    axis_labels = ['Vx', 'Vy', 'Vz']
    colors = {
        'fit_mean': '#3498db',      # 拟合曲线的蓝色
        'fit_ci': '#a9cce3',        # 置信区间的淡蓝色
        'predict_mean': '#e74c3c',  # 预测点的红色
        'predict_ci': '#f5b7b1',    # 预测误差棒的淡红色
        'data_points': '#2c3e50'   # 数据点的深灰色
    }

    # --- 3. 绘制每个维度的子图 ---
    for i in range(num_dims):
        ax = axes[i, 0]
        gp = online_gp_manager.gps[i]
        
        if not gp.is_trained_once:
            ax.set_title(f"Dimension {i} ({axis_labels[i]}) - Not Trained Yet", style='italic', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.6)
            continue

        # a) 提取、反归一化并绘制当前缓冲区中的数据点
        train_x_norm, train_y_norm = gp.get_and_normalize_data()
        train_x_denorm, train_y_denorm = None, None
        
        if train_x_norm is not None and train_y_norm is not None and train_x_norm.shape[0] > 0:
            v_mean_ema, v_std_ema = gp.v_mean_ema, np.sqrt(gp.v_var_ema + gp.epsilon)
            r_mean_ema, r_std_ema = gp.r_mean_ema, np.sqrt(gp.r_var_ema + gp.epsilon)
            train_x_denorm = train_x_norm.cpu().numpy().flatten() * v_std_ema + v_mean_ema
            train_y_denorm = train_y_norm.cpu().numpy().flatten() * r_std_ema + r_mean_ema
            
            # 绘制数据点，用 zorder 确保它在最上层
            ax.plot(train_x_denorm, train_y_denorm, 'o', color=colors['data_points'], 
                    markersize=5, label='Current Training Points', zorder=10)

        # b) 绘制GP在训练数据范围内的拟合曲线
        if train_x_denorm is not None and train_x_denorm.size > 0:
            x_min_plot, x_max_plot = train_x_denorm.min(), train_x_denorm.max()
            range_ext = (x_max_plot - x_min_plot) * 0.1
            if abs(range_ext) < 1e-4: range_ext = 0.5
            
            x_dense_denorm = np.linspace(x_min_plot - range_ext, x_max_plot + range_ext, 200)
            
            query_points_fit = np.zeros((len(x_dense_denorm), num_dims))
            # 为其他维度填充均值以获得准确的切片预测
            for j in range(num_dims):
                query_points_fit[:, j] = x_dense_denorm if i == j else online_gp_manager.gps[j].v_mean_ema

            fit_means, fit_vars = online_gp_manager.predict(query_points_fit)
            fit_mean_dim_i = fit_means[:, i]
            fit_std_dim_i = np.sqrt(np.maximum(fit_vars[:, i], 1e-9))

            # 绘制拟合曲线和置信区间
            ax.plot(x_dense_denorm, fit_mean_dim_i, color=colors['fit_mean'], lw=2.5, 
                    label='GP Fit (Training Domain)')
            ax.fill_between(x_dense_denorm, fit_mean_dim_i - 1.96 * fit_std_dim_i, fit_mean_dim_i + 1.96 * fit_std_dim_i,
                            color=colors['fit_ci'], alpha=0.6, label='95% CI (Training Domain)')

        # c) 绘制MPC规划点的预测结果
        pred_inputs_dim_i = future_velocities_np[:, i]
        mean_pred_dim_i = mpc_plan_means[:, i]
        std_pred_dim_i = np.sqrt(np.maximum(mpc_plan_vars[:, i], 1e-9))
        
        # 将MPC规划的预测点绘制为带误差棒的散点
        ax.errorbar(pred_inputs_dim_i, mean_pred_dim_i, yerr=1.96 * std_pred_dim_i,
                    fmt='o', color=colors['predict_mean'], markerfacecolor='white', markeredgecolor=colors['predict_mean'],
                    ecolor=colors['predict_ci'], elinewidth=2, capsize=3, mew=1.5,
                    ms=7, label='GP Prediction (MPC Plan)', zorder=5)

        # d) 图表美化
        ax.set_title(f"Dimension {i}: Body Velocity {axis_labels[i]} vs. Accel. Residual", fontsize=16)
        ax.set_xlabel(f"Body Velocity Component {axis_labels[i]} (m/s)", fontsize=12)
        ax.set_ylabel("Acceleration Residual (m/s^2)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    # 使用非阻塞模式显示，避免暂停仿真
    plt.show(block=False)
    plt.pause(0.1) # 短暂停顿以确保图形刷新
