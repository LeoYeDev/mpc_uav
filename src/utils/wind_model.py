"""
Wind model for UAV simulation.

This module provides realistic wind field models for MPC-UAV experiments.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from src.visualization.style import set_publication_style
from config.configuration_parameters import DirectoryConfig, SimpleSimConfig


class RealisticWindModel:
    """
    一个更符合物理现实的风场模型，基于多正弦波叠加。
    它模拟了一个缓慢变化的主风场，并叠加了多个频率和振幅不同的阵风分量。
    """
    def __init__(self, profile="default"):
        """
        定义风场模型的参数。
        - base_wind: 定义了缓慢变化的主风。
        - gusts: 一个列表，定义了多个快速变化的阵风/湍流分量。
        """
        wind_vel_params = {
            # 新增：风速随时间线性增长的斜率
            'ramp_slope': np.array([0.1, 0.1, 0.01]), # 各轴风速每秒增加量 (m/s^2)

            # 主风场：振幅减小，代表更平稳的整体趋势
            'base_wind': {
                'amp': np.array([0.2, 0.2, 0.05]),    # 各轴主风速振幅 (m/s) - 减小
                'freq': np.array([0.04, 0.03, 0.1]), # 各轴主风速变化频率 (rad/s) - 保持慢速
                'phase': np.array([0, np.pi/2, np.pi]), # 各轴风速相位
                'offset': np.array([1.5, 2.5, 0.2])  # 各轴风速初始偏置 (m/s) - 减小
            },
            # 阵风/湍流：振幅减小，数量减少，代表更小的波动
            'gusts': [
                {'amp': np.array([0.05, 0.05, 0.05]), 'freq': np.array([2.2, 2.9, 1.5]), 'phase': np.array([0.1, 1.5, 3.0])}, # 振幅减小
                {'amp': np.array([0.1, 0.15, 0.02]), 'freq': np.array([3.5, 3.1, 4.0]), 'phase': np.array([0.5, 2.5, 1.0])}, # 振幅减小
                # 移除了最高频的阵风分量以减少整体波动
            ]
        }
        self.params = wind_vel_params
        self.profile = str(profile)
        print(f"💨 [高级风场] 多正弦波叠加风场模型已初始化。")
        print(f"    - 风场模式 (Profile): {self.profile}")
        print(f"    - 初始偏置 (Offset): {self.params['base_wind']['offset']} m/s")
        print(f"    - 增长斜率 (Ramp Slope): {self.params.get('ramp_slope', np.zeros(3))} m/s²")
        print(f"    - 主风振幅 (Base Amp): {self.params['base_wind']['amp']} m/s")
        print(f"    - 主风频率 (Base Freq): {self.params['base_wind']['freq']} rad/s")
        print(f"    - 阵风分量数量: {len(self.params['gusts'])}")

    def get_wind_velocity(self, t):
        """根据时间 t 获取世界坐标系下的总风速向量。"""
        if self.profile == "regime_shift":
            return self._get_regime_shift_wind(t)

        return self._get_default_wind(t)

    def _get_default_wind(self, t):
        """Original smooth wind profile."""
        # X轴风速: f(t) = 1.3 * arctan(t - 4) + 1.8 + 0.2 * sin(0.7 * t)
        wind_x = 1.3 * np.arctan(t - 4) + 2.0 + 0.2 * np.sin(0.7 * t)
        
        # Y轴风速: g(t) = -1.0 * arctan(t - 9) - 0.5 + 0.2 * sin(0.5 * t)
        wind_y = -1.0 * np.arctan(t - 9) - 0.5 + 0.2 * np.sin(0.5 * t)
        
        # Z轴风速 (未指定，设为0)
        wind_z = 0.6 + 0.05 * np.sin(0.1 * t) + 0.05 * np.sin(1.5 * t) + 0.02 * np.sin(4.0 * t)
            
        return np.array([wind_x, wind_y, wind_z])

    def _get_regime_shift_wind(self, t):
        """
        Stronger non-stationary profile with regime changes and burst gusts.
        Useful for stressing online adaptation quality.
        """
        s1 = np.tanh((t - 5.0) / 1.5)
        s2 = np.tanh((t - 12.0) / 1.8)
        burst = np.exp(-0.5 * ((t - 9.0) / 0.7) ** 2) - 0.85 * np.exp(-0.5 * ((t - 15.0) / 1.1) ** 2)

        wind_x = 1.6 + 0.8 * s1 + 0.9 * s2 + 0.25 * np.sin(0.9 * t) + 0.12 * np.sin(3.0 * t) + 0.45 * burst
        wind_y = -0.9 - 0.6 * s1 + 0.7 * s2 + 0.18 * np.sin(0.7 * t + 0.8) + 0.10 * np.sin(2.5 * t) - 0.30 * burst
        wind_z = 0.4 + 0.12 * np.sin(0.2 * t) + 0.08 * np.sin(1.8 * t) + 0.03 * np.sin(4.5 * t)

        return np.array([wind_x, wind_y, wind_z])

    def visualize(self, duration=20):
        """可视化风速模型在一段时间内的函数图像，将三轴风速绘制在同一张图中。"""
        set_publication_style(base_size=9)  # 设置专业的出版物风格

        t_span = np.linspace(0, duration, 500)
        wind_velocities = np.array([self.get_wind_velocity(t) for t in t_span])

        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        axis_labels, colors = ['X-axis', 'Y-axis', 'Z-axis'], ['#d62728', '#1f77b4', '#2ca02c']
        for i in range(3):
            ax.plot(t_span, wind_velocities[:, i], color=colors[i], linewidth=1.5, label=f'{axis_labels[i]}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [m/s]')
        ax.grid(True)

        #显示图例
        ax.legend(loc='upper right', frameon=True)
        fig.tight_layout()

        # 保存图像
        plt.savefig("wind_velocity_visualization.pdf", bbox_inches="tight")
        # plt.show() # Commented out show to avoid blocking in non-interactive environment
