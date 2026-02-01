"""
GP offline visualization functions.

Provides publication-quality plots for visualizing offline GP training results.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import set_publication_style


def plot_offline_gp_fit(gp_regressor, x_train, y_train, full_dataset_gp, 
                        cluster_to_plot, title, input_dim, output_dim):
    """
    Create a publication-quality plot for visualizing offline GP model fit.
    
    Args:
        gp_regressor: Trained npGPRegression instance
        x_train: Training input data (n_samples, n_features)
        y_train: Training target data (n_samples,)
        full_dataset_gp: GPDataset containing full data
        cluster_to_plot: Cluster ID to plot
        title: Plot title
        input_dim: Input dimension index
        output_dim: Output dimension index
    """
    set_publication_style(base_size=9)
    
    # Get cluster data
    x_all_cluster = full_dataset_gp.get_x(cluster=cluster_to_plot)
    x_feature_dim = 0
    
    # Create dense sample grid for smooth prediction curve
    x_min, x_max = x_all_cluster[:, x_feature_dim].min(), x_all_cluster[:, x_feature_dim].max()
    x_range_ext = (x_max - x_min) * 0.05
    x_dense = np.linspace(x_min - x_range_ext, x_max + x_range_ext, 300).reshape(-1, 1)
    
    # Get GP predictions
    y_mean, y_std = gp_regressor.predict(x_dense, return_std=True)
    if y_std.ndim > 1:
        y_std = np.sqrt(np.diag(y_std))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    
    # Training points
    ax.scatter(x_train[:, x_feature_dim], y_train,
               s=25, facecolors='w', edgecolors='#3498db', 
               linewidth=1, zorder=10, label='Training Points')
    
    # GP mean curve
    ax.plot(x_dense, y_mean, color="#e74c3c", lw=1, zorder=5, label='GP Mean Fit')
    
    # 95% confidence interval
    ax.fill_between(x_dense.ravel(),
                    y_mean - 1.96 * y_std,
                    y_mean + 1.96 * y_std,
                    color="#e74c3c", alpha=0.2, label='95% CI')
    
    # Labels
    axis_map = {7: 'x', 8: 'y', 9: 'z'}
    x_sub = axis_map.get(input_dim, f'd_{input_dim}')
    y_sub = axis_map.get(output_dim, f'd_{output_dim}')
    ax.set_xlabel(fr'$v_{{{x_sub}}}$ (m/s)')
    ax.set_ylabel(fr'$\Delta a_{{{y_sub}}}$ (m/s²)')
    ax.set_xlim(-14, 14)
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    fig.tight_layout()
    plt.show()


def plot_gp_regression(x_test, y_test, x_train, y_train, gp_mean, gp_std, 
                       gp_regressor, labels, title='', n_samples=3):
    """
    Plot GP regression results with samples.
    
    Args:
        x_test: Test input points
        y_test: Test target values (optional, can be None)
        x_train: Training input points
        y_train: Training target values
        gp_mean: GP mean predictions
        gp_std: GP standard deviation predictions
        gp_regressor: GP regressor for sampling (optional, can be None)
        labels: Axis labels
        title: Plot title
        n_samples: Number of GP samples to draw
    """
    if len(x_test.shape) == 1:
        x_test = np.expand_dims(x_test, 1)
        n_subplots = 1
    else:
        n_subplots = x_test.shape[1]
    
    assert len(labels) == x_test.shape[1]
    
    # Generate samples if regressor provided
    y_samples = None
    if gp_regressor is not None:
        y_samples = gp_regressor.sample_y(x_test, n_samples)
        y_samples = np.squeeze(y_samples)
    
    for i in range(n_subplots):
        plt.subplot(n_subplots, 1, i + 1)
        
        # Sort by x values
        x_sort_ind = np.argsort(x_test[:, i])
        
        # GP mean
        plt.plot(x_test[x_sort_ind, i], gp_mean[x_sort_ind], 'k', lw=3, zorder=9)
        
        # GP uncertainty
        plt.fill_between(x_test[x_sort_ind, i],
                         gp_mean[x_sort_ind] - 3 * gp_std[x_sort_ind],
                         gp_mean[x_sort_ind] + 3 * gp_std[x_sort_ind],
                         alpha=0.2, color='k')
        
        # Samples
        if y_samples is not None:
            plt.plot(x_test[x_sort_ind, i], y_samples[x_sort_ind], '-o', lw=1)
            plt.xlim(min(x_test[:, i]), max(x_test[:, i]))
            plt.ylim(min(np.min(y_samples), np.min(gp_mean - 3 * gp_std)),
                     max(np.max(y_samples), np.max(gp_mean + 3 * gp_std)))
        
        # Training data
        if x_train is not None and y_train is not None:
            plt.scatter(x_train[:, i], y_train, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
        
        # Test data
        if y_test is not None:
            plt.plot(x_test[x_sort_ind, i], y_test[x_sort_ind], lw=1, marker='o')
        
        if i == 0 and title:
            plt.title(title, fontsize=12)
        
        plt.ylabel(labels[i])
    
    plt.tight_layout()
