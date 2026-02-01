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
    多正弦波叠加风场模型。
    
    模拟缓慢变化的主风场，并叠加多个频率和振幅不同的阵风分量。
    
    Attributes:
        params: 风场参数配置字典
    """
    
    def __init__(self, config=None):
        """
        初始化风场模型。
        
        Args:
            config: 可选的自定义配置字典。如果为None，使用默认配置。
        """
        if config is None:
            config = self._get_default_config()
        
        self.params = config
        self._log_init()
    
    def _get_default_config(self):
        """返回默认风场配置。"""
        return {
            'ramp_slope': np.array([0.1, 0.1, 0.01]),
            'base_wind': {
                'amp': np.array([0.2, 0.2, 0.05]),
                'freq': np.array([0.04, 0.03, 0.1]),
                'phase': np.array([0, np.pi/2, np.pi]),
                'offset': np.array([1.5, 2.5, 0.2])
            },
            'gusts': [
                {'amp': np.array([0.05, 0.05, 0.05]), 'freq': np.array([2.2, 2.9, 1.5]), 'phase': np.array([0.1, 1.5, 3.0])},
                {'amp': np.array([0.1, 0.15, 0.02]), 'freq': np.array([3.5, 3.1, 4.0]), 'phase': np.array([0.5, 2.5, 1.0])},
            ]
        }
    
    def _log_init(self):
        """打印初始化信息。"""
        print(f"💨 [Wind Model] Multi-sinusoid wind model initialized.")
        print(f"    - Offset: {self.params['base_wind']['offset']} m/s")
        print(f"    - Ramp Slope: {self.params.get('ramp_slope', np.zeros(3))} m/s²")
        print(f"    - Base Amplitude: {self.params['base_wind']['amp']} m/s")
        print(f"    - Gust components: {len(self.params['gusts'])}")
    
    def get_wind_velocity(self, t):
        """
        获取指定时间的风速向量（世界坐标系）。
        
        使用简化的arctan+sin组合模型。
        
        Args:
            t: 时间 (秒)
            
        Returns:
            np.ndarray: 3D风速向量 [vx, vy, vz] (m/s)
        """
        wind_x = 1.0 + 0.03 * np.sin(0.6 * t)
        wind_y = -0.4 + 0.01 * np.sin(0.5 * t)
        wind_z = 0.06 + 0.01 * np.sin(0.5 * t)
        return np.array([wind_x, wind_y, wind_z])
    
    def get_wind_velocity_full(self, t):
        """
        使用完整多正弦波模型计算风速。
        
        Args:
            t: 时间 (秒)
            
        Returns:
            np.ndarray: 3D风速向量 [vx, vy, vz] (m/s)
        """
        p = self.params
        base = p['base_wind']
        ramp_effect = p.get('ramp_slope', np.zeros(3)) * t
        wind_velocity = base['offset'] + ramp_effect + base['amp'] * np.sin(base['freq'] * t + base['phase'])
        
        for gust in p['gusts']:
            wind_velocity += gust['amp'] * np.sin(gust['freq'] * t + gust['phase'])
        
        return wind_velocity
    
    def visualize(self, duration=20, save=True):
        """
        可视化风速模型。
        
        Args:
            duration: 可视化时长 (秒)
            save: 是否保存图像
        """
        set_publication_style(base_size=9)
        
        t_span = np.linspace(0, duration, 500)
        wind_velocities = np.array([self.get_wind_velocity(t) for t in t_span])
        
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        axis_labels = ['X-axis', 'Y-axis', 'Z-axis']
        colors = ['#FD763F', '#23BAC5', '#EECA40']
        
        for i in range(3):
            ax.plot(t_span, wind_velocities[:, i], color=colors[i], linewidth=1.25, label=axis_labels[i])
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [m/s]')
        ax.grid(True)
        ax.legend(loc='upper right', frameon=True)
        fig.tight_layout()
        
        if save:
            fig_path = os.path.join(DirectoryConfig.FIGURES_DIR, 'wind_velocity_visualization')
            plt.savefig(fig_path + '.pdf', bbox_inches="tight")
            plt.savefig(fig_path + '.svg', bbox_inches="tight")
        
        if SimpleSimConfig.show_intermediate_plots:
            plt.show()
        else:
            plt.close()
