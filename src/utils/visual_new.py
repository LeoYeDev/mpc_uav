import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
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
#绘制跟踪结果
def plot_combined_results(combined_plot_data):
    set_publication_style(base_size=9)
    
    n = len(combined_plot_data)
    fig, axes = plt.subplots(1, n, figsize=(7.16, 2.8), sharey=True)
    axes = axes if n > 1 else [axes]

    # 统一误差范围与配色 - 使用更科学的配色方案
    all_err = np.hstack([
        np.linalg.norm(d['x_executed'][:, :2] - d['x_ref'][:, :2], axis=1)
        for d in combined_plot_data
    ])
    
    # 使用95%分位数作为最大值，避免极端值影响配色
    vmax = np.percentile(all_err, 95)
    norm = Normalize(vmin=0, vmax=vmax)
    
    # === 专业配色方案选择 ===
    # 方案1: 科学级配色 (Diverging)
    # cmap = plt.get_cmap('coolwarm')  # 蓝-红渐变
    # cmap = plt.get_cmap('RdYlBu_r')  # 红黄蓝反转 (红-低误差, 蓝-高误差)
    
    # 方案2: 单色渐变 (Sequential)
    # cmap = plt.get_cmap('viridis')  # 原始方案
    # cmap = plt.get_cmap('plasma')   # 紫色-黄色渐变
    cmap = plt.get_cmap('inferno')    # 黑色-黄色渐变 (高对比度)
    
    # 方案3: 自定义配色 (Perceptually Uniform)
    # 创建自定义感知均匀的配色方案
    colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]  # 绿-黄-橙-红
    cmap = LinearSegmentedColormap.from_list("custom_div", colors)
    
    # 方案4: 色盲友好配色
    # cmap = plt.get_cmap('cividis')  # 色盲友好方案
    # =========================

    for ax, data in zip(axes, combined_plot_data):
        t_ref, x_ref, x_exe = data['t_ref'], data['x_ref'], data['x_executed']
        title, wind_model = data['title'], data['wind_model']

        # 1) 参考轨迹 - 使用更细的虚线
        ax.plot(x_ref[:,0], x_ref[:,1], '--', color='#333333', lw=0.8, alpha=0.7, label='_nolegend_')

        # 2) 执行轨迹（误差着色）- 使用更精细的线段
        err = np.linalg.norm(x_exe[:, :2] - x_ref[:, :2], axis=1)
        pts = x_exe[:, :2].reshape(-1,1,2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        
        # 使用更精细的线段宽度和抗锯齿
        lc = LineCollection(
            segs, 
            cmap=cmap, 
            norm=norm, 
            linewidth=1.8,  # 略微加粗以突出颜色
            alpha=0.95,     # 轻微透明度
            antialiaseds=True,  # 启用抗锯齿
            label='Executed'
        )
        lc.set_array(err[:-1])
        ax.add_collection(lc)

        # 3) 风向箭头 - 使用更协调的颜色
        if wind_model is not None:
            idx = np.linspace(0, len(t_ref)-1, 15, dtype=int)  # 减少箭头数量
            pos = x_exe[idx, :2]
            wv = np.vstack([wind_model.get_wind_velocity(t)[:2] for t in t_ref[idx]])
            
            # 使用与主配色协调的箭头颜色
            arrow_color = '#23BAC5' if cmap.name == 'custom_div' else '#4c72b0'
            
            ax.quiver(
                pos[:,0], pos[:,1], wv[:,0], wv[:,1],
                color=arrow_color, 
                scale=50, 
                width=0.006,      # 更细的箭头
                headwidth=3.5,     # 更小的箭头头部
                headlength=4.5,
                label='Wind'
            )

        # 4) 标题样式优化
        ax.set_title(f"{title}", fontsize=7, pad=5)  # 增加上边距
        
        ax.set_xlabel(r'$p_x$ [m]', labelpad=3)  # 增加标签间距
        if ax is axes[0]:
            ax.set_ylabel(r'$p_y$ [m]', labelpad=5)
        
        # 优化网格和坐标轴
        ax.grid(True)  # 虚线网格
        
        # 添加轻微边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_color('#cccccc')

    # 统一 colorbar - 优化样式
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = plt.colorbar(lc, cax=cax)
    cb.set_label('Position Error [m]', rotation=270, labelpad=15, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    
    # 统一 legend - 优化样式
    fig.tight_layout()
    legend_handles = [
        Line2D([0],[0], linestyle='--', color='#333333', lw=1, alpha=0.7, label='Ref'),
        Line2D([0],[0], color=cmap(0.5), lw=3, label='Executed'),  # 使用配色中间值
        Line2D([0],[0], marker=r'$\rightarrow$', color=arrow_color,
               linestyle='None', markersize=10, label='Wind')
    ]
    fig.legend(
        legend_handles, 
        ['Ref','Executed','Wind'],
        loc='upper center', 
        bbox_to_anchor=(0.5,0.98),
        ncol=3, 
        frameon=True
    )

    # 整体布局优化
    plt.subplots_adjust(top=0.83, wspace=0.1, right=0.9)
    
    # 保存多种格式
    plt.savefig("combined_tracking_results.pdf", bbox_inches="tight", dpi=600)
    plt.savefig("combined_tracking_results.svg", bbox_inches="tight", dpi=600)
    
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
    plt.savefig("tracking_error_comparison.svg", bbox_inches="tight")
    plt.show()

