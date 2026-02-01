"""
Visualization module for MPC-UAV project.

Provides plotting, animation, and style utilities for research visualization.
"""

from src.visualization.style import set_publication_style, SCI_COLORS
from src.visualization.plotting import (
    initialize_drone_plotter,
    draw_drone_simulation,
    trajectory_tracking_results,
    mse_tracking_experiment_plot,
    visualize_data_distribution,
    visualize_gp_inference
)
from src.visualization.paper_plots import (
    plot_combined_results,
    plot_tracking_error_comparison,
    tracking_results_with_wind
)
from src.visualization.animation import Dynamic3DTrajectory

__all__ = [
    # Style
    'set_publication_style',
    'SCI_COLORS',
    # Plotting
    'initialize_drone_plotter',
    'draw_drone_simulation',
    'trajectory_tracking_results',
    'mse_tracking_experiment_plot',
    'visualize_data_distribution',
    'visualize_gp_inference',
    # Paper plots
    'plot_combined_results',
    'plot_tracking_error_comparison',
    'tracking_results_with_wind',
    # Animation
    'Dynamic3DTrajectory',
]
