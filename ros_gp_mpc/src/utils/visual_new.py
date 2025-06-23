import matplotlib.pyplot as plt
import numpy as np
from src.utils.visual_set import set_publication_style

def tracking_results(t_ref, x_ref, x_executed, u_ref, u_executed, title, 
                                w_control=None, legend_labels=None, quat_error=True):
    """
    创建一个专为学术论文优化的、展示单次实验轨迹跟踪性能的图表。
    此函数接口与旧版保持一致，但生成新的、更清晰的可视化结果。

    该图包含一个2D平面的轨迹对比图 (X-Y Plane)。

    :param t_ref: np.ndarray, 时间戳数组。
    :param x_ref: np.ndarray, 参考轨迹状态 [N, 13]。
    :param x_executed: np.ndarray, 实际执行轨迹状态 [N, 13]。
    :param u_ref, u_executed, w_control, legend_labels, quat_error: 为保持接口兼容而保留的参数，在此函数中未使用。
    :param title: 图表的总标题。
    :param save_path: 可选，如果提供路径，则将图表保存为文件。
    """
    
    # --- 设置专业的绘图风格 ---
    set_publication_style(font_size=16)  # 设置专业的出版物风格

    # --- 创建一个 1x1 的子图布局 ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)

    # --- 绘制2D平面轨迹图 (X-Y Plane) ---
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True, linestyle=':')
    ax.axis('equal')

    # 绘制参考轨迹和执行轨迹
    ax.plot(x_ref[:, 0], x_ref[:, 1], 
                 color='black', linestyle='--', linewidth=2, label='Reference Trajectory')
    ax.plot(x_executed[:, 0], x_executed[:, 1], 
                 color='royalblue', linestyle='-', linewidth=2.5, label='Executed Trajectory')
    ax.legend()

    # --- 设置总标题并调整布局 ---
    if title:
        fig.suptitle(title)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # --- 显示 ---
    plt.show()


# ==============================================================================
# --- 新增函数：用于生成论文级的多控制器跟踪误差对比图 ---
# ==============================================================================
def plot_tracking_error_comparison(results_data, controller_map, title=""):
    """
    从内存中存储的多个仿真结果中加载数据，并绘制一个用于论文的跟踪误差对比图。
    该图的风格模仿了您提供的参考图片。

    :param results_data: 一个字典，键是控制器名称，值是包含仿真结果的另一个字典。
                         例如: {'DGP-MPC': {'t_ref': array, 'x_ref': array, 'x_executed': array}, ...}
    :param controller_map: 一个字典，用于定义每个控制器的绘图样式。
                           例如: {'DGP-MPC': {'color': 'purple', 'linestyle': '-', 'label': 'DGP-MPC'}, ...}
    :param title: 图表的总标题。
    """
    
    # --- 1. 设置专业的绘图风格 (与参考图保持一致) ---
    set_publication_style(font_size=16)  # 设置专业的出版物风格

    # --- 2. 创建 3x1 的子图布局 (分别对应 X, Y, Z 轴误差) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=150)
    
     # --- 3. 循环加载每个控制器的结果并绘图 ---
    for controller_name, data in results_data.items():
        # 直接从传入的字典中获取数据
        t = data['t_ref']
        x_ref = data['x_ref']
        x_exec = data['x_executed']
        
        # 获取绘图样式
        style = controller_map.get(controller_name, {})
        color = style.get('color', 'black')
        linestyle = style.get('linestyle', '-')
        linewidth = style.get('linewidth', 2)
        label = style.get('label', controller_name)
        fill_alpha = style.get('fill_alpha', 0.15) # 用于填充阴影区域的透明度
        zorder = style.get('zorder', 1) # 从样式中获取zorder，默认为1

        # 计算位置误差
        pos_error = np.abs(x_exec[:, :3] - x_ref[:, :3])
        
        # 在每个子图上绘制对应轴的误差曲线
        for i in range(3):
            ax = axes[i]
            # 绘制误差曲线，zorder比阴影高一点，确保线在阴影之上
            ax.plot(t, pos_error[:, i], 
                    color=color, 
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label,
                    zorder=zorder + 1) # 将线绘制在阴影之上
            # 绘制阴影区域
            if fill_alpha > 0:
                ax.fill_between(t, 0, pos_error[:, i], color=color, alpha=fill_alpha, zorder=zorder)


    # --- 4. 美化图表 ---
    axis_labels = ['Positional Error-X (m)', 'Positional Error-Y (m)', 'Positional Error-Z (m)']
    for i, ax in enumerate(axes):
        ax.set_ylabel(axis_labels[i])
        ax.grid(True, linestyle=':', which='both')
        # 设置Y轴的下限为0，让图形更美观
        ax.set_ylim(bottom=0)
        # 优化Y轴刻度范围，避免数据贴近上边缘
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1) 

    # 设置共享的X轴标签
    axes[-1].set_xlabel('Time (s)')

    # --- 5. 创建位于顶部的共享图例 ---
    # 从第一个子图获取所有曲线的句柄和标签
    handles, labels = axes[0].get_legend_handles_labels()
    # 通过字典去重，以防同一个控制器被多次添加图例
    by_label = dict(zip(labels, handles))
    # 在图表顶部创建图例
    fig.legend(by_label.values(), by_label.keys(),
               loc='upper center',      # 定位在顶部中央
               bbox_to_anchor=(0.5, 1.0), # 精确控制位置在顶部
               ncol=len(by_label),      # 横向排列
               frameon=True,  # 显示边框
               fontsize=15)           
    # --- 6. 设置总标题并调整布局 ---
    if title:
        fig.suptitle(title, fontsize=24, y=1.05) # 调整y值，为图例留出空间

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # 调整布局，防止元素重叠

    # --- 7. 显示 ---
    plt.show()

