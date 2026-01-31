import matplotlib.pyplot as plt
import numpy as np

# 为您的项目定义一套统一、专业的颜色方案
# 这是基于 Matplotlib 的 'tab10' 色板，非常清晰且对色盲友好
SCI_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', "#9e9d9d",
              '#bcbd22', '#17becf']

def set_publication_style(base_size=9):
    """
    为所有图表设置一个统一的、适合高水平学术期刊的专业风格。
    此函数修改全局的 plt.rcParams，在任何绘图函数之前调用即可生效。
    :param base_size: int, 图表的基础字号，其他元素字号会依此调整。
    """
    try:
        # 使用一个干净的、适合论文的基础风格
        #plt.style.use('seaborn-paper')
        
        # 应用更精细的期刊样式配置
        plt.rcParams.update({
            # 基础设置保持不变
            'font.size': base_size,
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
            'mathtext.fontset': 'stix',

            # 关键调整：坐标轴和刻度线变细
            'axes.linewidth': 0.5,         # 坐标轴边框粗细（原0.8→0.5）
            'xtick.major.width': 0.5,      # x轴主刻度线粗细（原0.8→0.5）
            'ytick.major.width': 0.5,      # y轴主刻度线粗细（原0.8→0.5）
            'xtick.minor.width': 0.3,      # x轴次刻度线粗细（原默认值→0.3）
            'ytick.minor.width': 0.3,      # y轴次刻度线粗细（原默认值→0.3）

            # 刻度大小也可微调（可选）
            'xtick.major.size': 3,         # 主刻度长度（原4→3）
            'ytick.major.size': 3,         # 主刻度长度（原4→3）

            # 其他设置保持不变
            'axes.labelsize': base_size,
            'axes.titlesize': base_size + 1,
            'xtick.labelsize': base_size - 1,
            'ytick.labelsize': base_size - 1,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'legend.fontsize': base_size - 1,
            'legend.frameon': False,
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.5,
            'savefig.dpi': 600,
            'savefig.format': 'pdf',
            'figure.figsize': (3.5, 2.2),
        })
        # plt.rcParams.update({
        #     'font.size': base_size,
        #     'font.family': 'Times New Roman',
        #     'mathtext.fontset': 'stix',  # 让数学字体更接近Times
        #     'axes.labelsize': base_size,
        #     'axes.titlesize': base_size + 1,
        #     # 'axes.linewidth': 0.8,
        #     'axes.labelweight': 'normal',
        #     'axes.titleweight': 'normal',
        #     'xtick.labelsize': base_size - 1,
        #     'ytick.labelsize': base_size - 1,
        #     'xtick.direction': 'in',
        #     'ytick.direction': 'in',
        #     'xtick.major.width': 0.8,
        #     'ytick.major.width': 0.8,
        #     'xtick.major.size': 4,
        #     'ytick.major.size': 4,
        #     'xtick.minor.size': 2,
        #     'ytick.minor.size': 2,
        #     'legend.fontsize': base_size - 1,
        #     'legend.frameon': False,
        #     'grid.linestyle': '--',
        #     'grid.linewidth': 0.5,
        #     'grid.alpha': 0.5,
        #     'savefig.dpi': 600,
        #     'savefig.format': 'pdf',
        #     'figure.figsize': (3.5, 2.2),  # 单栏图推荐尺寸（英寸）
            
        # })
    except Exception as e:
        print(f"⚠️ 设置绘图风格失败")
if __name__ == "__main__":
    # 测试 set_publication_style 函数
    set_publication_style(base_size=9)

    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 创建单栏图
    fig, ax = plt.subplots()  # 单栏图尺寸
    ax.plot(x, y1, label="sin(x)", color=SCI_COLORS[0], linewidth=1.5)
    ax.plot(x, y2, label="cos(x)", color=SCI_COLORS[1], linewidth=1.5)

    # 设置轴标签和图例
    ax.set_xlabel("X-axis [unit]")
    ax.set_ylabel("Y-axis [unit]")
    ax.legend(loc="upper right")

    # 显示网格
    ax.grid(True)

    # 保存图片并显示
    plt.savefig("test_publication_style.pdf", bbox_inches="tight")
    plt.show()