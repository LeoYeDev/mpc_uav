import matplotlib.pyplot as plt
import numpy as np

# 为您的项目定义一套统一、专业的颜色方案
# 这是基于 Matplotlib 的 'tab10' 色板，非常清晰且对色盲友好
SCI_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', "#9e9d9d",
              '#bcbd22', '#17becf']

def set_publication_style(base_size=11):
    """
    为所有图表设置一个统一的、适合高水平学术期刊的专业风格。
    此函数修改全局的 plt.rcParams，在任何绘图函数之前调用即可生效。

    该风格特点:
    - 字体: 使用清晰的无衬线字体 (Arial/Helvetica style)。
    - 简洁: 无背景色，轻量级网格线，无图例边框。
    - 清晰: 坐标轴刻度朝内，线宽和标记大小适中。
    - 专业: 字体大小比例协调，适合嵌入论文。

    :param base_size: int, 图表的基础字号，其他元素字号会依此调整。
    """
    try:
        # 使用一个干净的、适合论文的基础风格
        plt.style.use('seaborn-paper')
        
        # 应用更精细的SCI期刊级样式配置
        plt.rcParams.update({
            # --- 字体设置 (使用清晰的无衬线字体) ---
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], # 字体栈
            'mathtext.fontset': 'dejavusans', # 数学符号使用匹配的无衬线字体
            
            # --- 全局字号与粗细 ---
            'font.size': base_size,
            'font.weight': 'normal', # 除非特别指定，否则使用正常粗细
            'axes.labelweight': 'normal',
            'axes.titleweight': 'bold', # 仅将标题设为粗体

            # --- 坐标轴与刻度 ---
            'axes.linewidth': 1.0,              # 坐标轴线宽
            'xtick.direction': 'in',            # X轴刻度朝内
            'ytick.direction': 'in',            # Y轴刻度朝内
            'xtick.major.width': 1.0,           # X轴主刻度线宽
            'ytick.major.width': 1.0,           # Y轴主刻度线宽
            'xtick.major.size': 5.0,            # X轴主刻度长度
            'ytick.major.size': 5.0,            # Y轴主刻度长度
            'xtick.minor.width': 0.8,           # X轴次刻度线宽
            'ytick.minor.width': 0.8,           # Y轴次刻度线宽
            'xtick.minor.size': 3.0,            # X轴次刻度长度
            'ytick.minor.size': 3.0,            # Y轴次刻度长度
            'xtick.minor.visible': True,        # 默认显示次刻度

            # --- 元素字号比例 ---
            'axes.titlesize': base_size + 2,    # 坐标轴标题字号
            'axes.labelsize': base_size,        # 坐标轴标签字号
            'legend.fontsize': base_size - 2,   # 图例字号
            'xtick.labelsize': base_size - 2,   # X轴刻度标签字号
            'ytick.labelsize': base_size - 2,   # Y轴刻度标签字号
            
            # --- 图例与网格 ---
            'legend.frameon': False,            # 图例无边框
            'grid.linestyle': '--',             # 网格线样式
            'grid.linewidth': 0.7,              # 网格线线宽
            'grid.alpha': 0.7,                  # 网格线透明度

            # --- 保存图像的默认设置 ---
            'savefig.dpi': 300,                 # 高分辨率
            'savefig.format': 'pdf',            # 优先保存为矢量图
        })
        print(f"✅ 已成功应用SCI期刊级绘图风格 (基础字号: {base_size})。")
    except Exception as e:
        print(f"⚠️ 设置绘图风格失败: {e}。将使用Matplotlib默认风格。")