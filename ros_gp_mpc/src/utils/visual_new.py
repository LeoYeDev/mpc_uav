import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from src.utils.visual_set import set_publication_style

global combined_plot_data
combined_plot_data = []

def tracking_results_with_wind(t_ref, x_ref, x_executed, title, wind_model=None):
    """
    创建一个高级的2D轨迹跟踪可视化图，专为论文设计，以突出风场影响。

    该图包含:
    1. 参考轨迹与实际执行轨迹。
    2. 实际执行轨迹根据跟踪误差进行颜色编码。
    3. (可选) 在轨迹上用箭头可视化随时间变化的风场。
    4. 一个颜色条图例，解释误差大小。

    :param t_ref: np.ndarray, 时间戳数组。
    :param x_ref: np.ndarray, 参考轨迹状态 [N, 13]。
    :param x_executed: np.ndarray, 实际执行轨迹状态 [N, 13]。
    :param title: str, 图表的总标题。
    :param wind_model: RealisticWindModel (可选), 用于获取风速的风场模型实例。
    """
    # 保存数据到全局变量
    combined_plot_data.append({
        't_ref': t_ref,
        'x_ref': x_ref,
        'x_executed': x_executed,
        'title': title,
        'wind_model': wind_model
    })
    # # 1. 设置专业的绘图风格
    # set_publication_style(base_size=9)

    # fig, ax = plt.subplots(figsize=(9, 7))

    # # 2. 绘制参考轨迹
    # ax.plot(x_ref[:, 0], x_ref[:, 1],
    #         color='black',
    #         linestyle='--',
    #         linewidth=1.0,
    #         label='Reference Trajectory')

    # # 3. 计算瞬时跟踪误差
    # # 注意: 确保参考轨迹和执行轨迹有相同的长度，如果不同需要先插值。
    # # 这里我们假设它们是对齐的。
    # error = np.linalg.norm(x_executed[:, :2] - x_ref[:, :2], axis=1)

    # # 4. 创建用于颜色编码的 LineCollection
    # points = x_executed[:, :2].reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # # 将误差值归一化到 [0, 1] 用于着色
    # norm = Normalize(vmin=np.min(error), vmax=np.max(error))
    # cmap = plt.get_cmap('viridis') # 使用 'plasma', 'viridis' 或 'hot' 颜色图效果很好
    
    # lc = LineCollection(segments, cmap=cmap, norm=norm)
    # # 设置数组，颜色将基于每个线段起点的误差值
    # lc.set_array(error[:-1])
    # lc.set_linewidth(3.0) # 设置轨迹线宽
    # line = ax.add_collection(lc)
    # # 手动为 LineCollection 添加图例标签（它本身不直接支持label）
    # lc.set_label('Executed Trajectory (Colored by Error)')


    # # 5. 添加颜色条 (Colorbar)作为误差图例
    # cbar = fig.colorbar(line, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    # cbar.set_label('Position Error (m)', rotation=270, labelpad=25)
    
    # # 6. (可选) 可视化风场
    # wind_proxy_handle = None
    # if wind_model is not None:
    #     # 在轨迹上选择N个点来绘制风矢量
    #     num_arrows = 20
    #     indices = np.linspace(0, len(t_ref) - 1, num_arrows, dtype=int)
        
    #     # 获取这些点的无人机位置和时间
    #     arrow_positions = x_executed[indices, :2]
    #     arrow_times = t_ref[indices]
        
    #     # 获取对应的风速 (只取XY分量)
    #     wind_vectors = np.array([wind_model.get_wind_velocity(t)[:2] for t in arrow_times])
        
    #     # 绘制风箭头 (Quiver plot)
    #     wind_color = '#2ca02c' # 使用绿色表示风
    #     ax.quiver(arrow_positions[:, 0], arrow_positions[:, 1],
    #               wind_vectors[:, 0], wind_vectors[:, 1],
    #               color=wind_color,
    #               scale=60,       # 缩放因子，可能需要根据您的风速大小进行调整
    #               width=0.004,    # 箭头宽度
    #               headwidth=4,    # 箭头头部宽度
    #               alpha=0.8)      # 透明度
        
    #     # 创建一个代理艺术家（proxy artist）用于在图例中清晰地显示风矢量
    #     wind_proxy_handle = Line2D([0], [0], color=wind_color, lw=0, 
    #                                marker=r'$\rightarrow$', # 使用箭头符号
    #                                markersize=15, 
    #                                label='Wind Vector')

    # # 7. 美化图表
    # ax.set_xlabel(r'$p_x$ (m)')
    # ax.set_ylabel(r'$p_y$ (m)')
    # ax.set_title(title)
    # ax.grid(True, linestyle=':', alpha=0.7)
    # ax.set_aspect('equal', adjustable='box') # 确保X和Y轴比例相同
    
    # # 8. 创建并显示最终的图例
    # handles, labels = ax.get_legend_handles_labels()
    
    # # 如果有风场，将风场图例句柄添加进去
    # if wind_proxy_handle:
    #     handles.append(wind_proxy_handle)

    # ax.legend(handles=handles, loc='best')
    
    # fig.tight_layout()
    # plt.show()
#绘制跟踪结果
def plot_combined_results(combined_plot_data):
    set_publication_style(base_size=9)

    n = len(combined_plot_data)
    fig, axes = plt.subplots(1, n, figsize=(7.16, 2.8), sharey=True)
    axes = axes if n > 1 else [axes]

    # 统一误差范围与配色
    all_err = np.hstack([
        np.linalg.norm(d['x_executed'][:, :2] - d['x_ref'][:, :2], axis=1)
        for d in combined_plot_data
    ])
    norm = Normalize(vmin=0, vmax=all_err.max())
    cmap = plt.get_cmap('viridis')

    for ax, data in zip(axes, combined_plot_data):
        t_ref, x_ref, x_exe = data['t_ref'], data['x_ref'], data['x_executed']
        title, wind_model    = data['title'], data['wind_model']

        # 1) 参考轨迹
        ax.plot(x_ref[:,0], x_ref[:,1], '--k', lw=1, label='_nolegend_')

        # 2) 执行轨迹（误差着色）
        err = np.linalg.norm(x_exe[:, :2] - x_ref[:, :2], axis=1)
        pts = x_exe[:, :2].reshape(-1,1,2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2, label='Executed')
        lc.set_array(err[:-1])
        ax.add_collection(lc)

        # 3) 风向箭头
        if wind_model is not None:
            idx = np.linspace(0, len(t_ref)-1, 20, dtype=int)
            pos = x_exe[idx, :2]
            wv  = np.vstack([wind_model.get_wind_velocity(t)[:2] for t in t_ref[idx]])
            ax.quiver(pos[:,0], pos[:,1], wv[:,0], wv[:,1],
                      color='#2ca02c', scale=50, width=0.008,
                      headwidth=4, headlength=4, label='Wind')

        # 4) 标题简化：控制器名 + 性能指标
        ax.set_title(f"{title}",fontsize=6)

        ax.set_xlabel(r'$p_x$ [m]')
        if ax is axes[0]:
            ax.set_ylabel(r'$p_y$ [m]')
        ax.grid(True)
        ax.set_aspect('equal', 'box')

    # 统一 colorbar
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = plt.colorbar(lc, cax=cax)
    cb.set_label('Position Error [m]', rotation=270, labelpad=15)
    cb.ax.tick_params(labelsize=7)

    # 统一 legend：Ref, Executed, Wind
    legend_handles = [
        Line2D([0],[0], linestyle='--', color='k', lw=1, label='Ref'),
        Line2D([0],[0], color='#37728e', lw=3, label='Executed'),
        Line2D([0],[0], marker=r'$\rightarrow$', color='#2ca02c',
               linestyle='None', markersize=10, label='Wind')
    ]
    fig.legend(legend_handles, ['Ref','Executed','Wind'],
               loc='upper center', bbox_to_anchor=(0.5,0.96),
               ncol=3, frameon=True, fontsize=8, handlelength=1.5)

    plt.subplots_adjust(top=0.86, wspace=0.07)
    plt.savefig("combined_tracking_results.pdf", bbox_inches="tight")
    plt.show()
# ==============================================================================
# --- 新增函数：用于生成论文级的多控制器跟踪误差对比图 ---
# ==============================================================================
def plot_tracking_error_comparison(results_data, controller_map):
    """
    从内存中存储的多个仿真结果中加载数据，并绘制一个用于论文的跟踪误差对比图。
    该图的风格模仿了您提供的参考图片。
    :param results_data: 一个字典，键是控制器名称，值是包含仿真结果的另一个字典。
                         例如: {'AR-MPC': {'t_ref': array, 'x_ref': array, 'x_executed': array}, ...}
    :param controller_map: 一个字典，用于定义每个控制器的绘图样式。
                           例如: {'AR-MPC': {'color': 'purple', 'linestyle': '-', 'label': 'ARMPC'}, ...}
    :param title: 图表的总标题。
    """
    # --- 1. 设置专业的绘图风格 (与参考图保持一致) ---
    set_publication_style(base_size=9)  # 设置专业的出版物风格

    # --- 2. 创建 3x1 的子图布局 (分别对应 X, Y, Z 轴误差) ---
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(3.5, 1.2*3))  # 单栏图尺寸
    
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
    axis_labels = ['Error-X [m]', 'Error-Y [m]', 'Error-Z [m]']
    for i, ax in enumerate(axes):
        ax.set_ylabel(axis_labels[i])
        ax.grid(True)
        # 设置Y轴的下限为0，让图形更美观
        ax.set_ylim(bottom=0)
        # 优化Y轴刻度范围，避免数据贴近上边缘
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1) 
        from matplotlib.ticker import StrMethodFormatter
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))  # 设置为一位小数

    # 设置共享的X轴标签
    axes[-1].set_xlabel('Time [s]')

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
        hspace=0.1,  # 增加子图间距
        top=0.85     # 预留顶部空间给图例
    )

    plt.savefig("tracking_error_comparison.pdf", bbox_inches="tight")
    plt.show()

