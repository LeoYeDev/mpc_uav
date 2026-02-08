""" Set of tunable parameters for the Simplified Simulator and model fitting.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os


class DirectoryConfig:
    """
    存储项目目录结构的配置类
    Class for storing directories within the package
    """

    _dir_path = os.path.dirname(os.path.realpath(__file__))
    _root_path = os.path.dirname(_dir_path)
    
    # 新的统一输出目录
    # New consolidated outputs directory
    OUTPUTS_DIR = os.path.join(_root_path, 'outputs')
    MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')        # 模型保存目录
    EXPERIMENTS_DIR = os.path.join(OUTPUTS_DIR, 'experiments') # 实验结果保存目录
    FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')      # 图片保存目录
    ACADOS_CACHE_DIR = os.path.join(OUTPUTS_DIR, 'acados_cache') # ACADOS 缓存目录
    
    # 为向后兼容保留的旧别名
    # Legacy aliases for backwards compatibility
    SAVE_DIR = MODELS_DIR
    RESULTS_DIR = OUTPUTS_DIR
    CONFIG_DIR = _dir_path
    DATA_DIR = os.path.join(_root_path, 'data')             # 数据集目录


class SimpleSimConfig:
    """
    简易仿真器配置类
    Class for storing the Simplified Simulator configurations.
    """

    # 设置为 True 以显示简易仿真器的实时 Matplotlib 动画。
    # 开启 GUI 会使执行速度变慢。注意：设置为 True 可能需要安装额外的库。
    # Set to True to show a real-time Matplotlib animation of the experiments for the Simplified Simulator. Execution
    # will be slower if the GUI is turned on. Note: setting to True may require some further library installation work.
    custom_sim_gui = True

    # 设置为 True 以在执行后显示轨迹跟踪结果图表。
    # Set to True to display a plot describing the trajectory tracking results after the execution.
    result_plots = True

    # 设置为 True 以在执行时间之前显示将要执行的轨迹。
    # Set to True to show the trajectory that will be executed before the execution time
    pre_run_debug_plots = True
    
    # 设置为 False 以减少仿真过程中的 matplotlib 弹窗。
    # Set to False to reduce matplotlib popup windows during simulation
    show_intermediate_plots = False

    # 简易仿真器中建模的扰动类型选择。详细参数请参考脚本：src/quad_mpc/quad_3d.py。
    # Choice of disturbances modeled in our Simplified Simulator. For more details about the parameters used refer to
    # the script: src/quad_mpc/quad_3d.py.
    simulation_disturbances = {
        "noisy": True,                       # 推力和力矩的高斯噪声 (Thrust and torque gaussian noises)
        "drag": True,                        # 二阶多项式气动阻力效应 (2nd order polynomial aerodynamic drag effect)
        "payload": False,                    # Z 轴方向的载荷扰动 (Payload force in the Z axis)
        "motor_noise": True                  # 电机非对称电压噪声 (Asymmetric voltage noise in the motors)
    }


class ModelFitConfig:
    """
    模型拟合脚本标志配置类
    Class for storing flags for the model fitting scripts.
    """

    # ## 数据集加载 (Dataset loading) ## #
    ds_name = "simplified_sim"
    ds_metadata = {
        "noisy": True,
        "drag": True,
        "payload": False,
        "motor_noise": True
    }

    #ds_name = "gazebo_dataset"
    # ds_metadata = {
    #     "gazebo": "default",
    # }

    # ## 可视化 (Visualization) ## #
    # 训练模式 (Training mode)
    visualize_training_result = True
    visualize_data = True

    # 可视化模式 (Visualization mode)
    grid_sampling_viz = True  # 是否可视化网格采样
    x_viz = [7, 8, 9]         # 可视化的状态维度 (通常是速度)
    u_viz = []                # 可视化的输入维度
    y_viz = [7, 8, 9]         # 可视化的输出维度

    # ## 数据后处理 (Data post-processing) ## #
    histogram_bins = 40              # 使用直方图分箱聚类数据 (Cluster data using histogram binning)
    histogram_threshold = 0.001      # 移除数据比例低于此阈值的箱 (Remove bins where the total ratio of data is lower than this threshold)
    velocity_cap = 16                # 如果 abs(velocity) > x_cap，则移除该数据点 (Also remove datasets point if abs(velocity) > x_cap)

    # ############# 实验性功能 (Experimental) ############# #

    # ## 使用拟合模型生成合成数据 (Use fit model to generate synthetic data) ## #
    use_dense_model = False
    dense_model_version = ""
    dense_model_name = ""
    dense_training_points = 200

    # ## 多维模型聚类 (Clustering for multidimensional models) ## #
    clusters = 1
    load_clusters = False
