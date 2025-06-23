import matplotlib.pyplot as plt
import numpy as np

def set_publication_style(font_size=16):
    """
    为所有图表设置一个统一的、适合出版物的专业风格。
    通过修改 plt.rcParams，此函数会影响之后创建的所有图表。
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            # --- 字体设置 ---
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 14, # 稍微增大基础字号，提高可读性

            # --- 坐标轴与标题字号 ---
            "axes.titlesize": font_size + 2, # 坐标轴标题字号
            "axes.labelsize": font_size,     # 坐标轴标签字号
            
            # --- 刻度标签字号 ---
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,

            # --- 图例字号 ---
            "legend.fontsize": font_size - 2,
            
            # --- Figure标题字号 ---
            "figure.titlesize": font_size + 4,
        })
        print("✅ 已成功应用出版物级绘图风格。")
    except Exception as e:
        print(f"⚠️ 设置绘图风格失败: {e}。将使用Matplotlib默认风格。")
        plt.style.use('default')