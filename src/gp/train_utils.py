import torch
import gpytorch
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Import ExactGPModel from online.py (architecture unification)
from src.gp.online import ExactGPModel
from config.gp_config import GPModelParams

def train_gp_torch(
    train_x: np.ndarray, 
    train_y: np.ndarray, 
    n_iter: int = 100, 
    lr: float = 0.05, 
    device_str: str = 'cpu',
    verbose: bool = False
) -> Tuple[ExactGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    """
    Train a GP model using GPyTorch.
    
    Args:
        train_x: Input features (N, D)
        train_y: Target values (N,)
        n_iter: Number of training iterations
        lr: Learning rate
        device_str: 'cpu' or 'cuda'
        verbose: Print training progress
        
    Returns:
        model: Trained ExactGPModel
        likelihood: Trained GaussianLikelihood
    """
    device = torch.device(device_str)
    
    # Convert data to tensors
    train_x_tensor = torch.from_numpy(train_x).float().to(device)
    train_y_tensor = torch.from_numpy(train_y).float().to(device)
    
    # Initialize model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    # Correctly reshape input for ARD (Automatic Relevance Determination)
    # ExactGPModel constructor expects inputs to initialize shapes
    model = ExactGPModel(train_x_tensor, train_y_tensor, likelihood, ard_num_dims=train_x.shape[1]).to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    start_time = time.time()
    
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x_tensor)
        loss = -mll(output, train_y_tensor)
        loss.backward()
        optimizer.step()
        
        if verbose and (i % 10 == 0 or i == n_iter - 1):
            print(f"Iter {i+1}/{n_iter} - Loss: {loss.item():.3f}   lengthscale: {model.covar_module.base_kernel.lengthscale.mean().item():.3f}   noise: {likelihood.noise.item():.3f}")

    if verbose:
        print(f"Training completed in {time.time() - start_time:.2f}s")
        
    return model, likelihood

def extract_gp_params(model: ExactGPModel, likelihood: gpytorch.likelihoods.GaussianLikelihood) -> GPModelParams:
    """
    Extract parameters from trained GPyTorch model into standardized GPModelParams.
    """
    model.eval()
    likelihood.eval()
    
    # safely extract parameters to cpu numpy
    length_scale = model.covar_module.base_kernel.lengthscale.cpu().detach().view(-1).numpy().tolist()
    output_scale = model.covar_module.outputscale.cpu().detach().item()
    noise = likelihood.noise.cpu().detach().item()
    mean_val = model.mean_module.constant.cpu().detach().item()
    
    return GPModelParams(
        length_scale=length_scale,
        signal_variance=output_scale,
        noise_variance=noise,
        mean=mean_val
    )

def compute_matrices_for_casadi(
    params: GPModelParams, 
    train_x: np.ndarray, 
    train_y: np.ndarray,
    normalize_y: bool = False # Legacy param, usually handled outside
) -> Dict[str, Any]:
    """
    Compute K_inv and K_inv_y (alpha) using numpy for CasADi compatibility.
    This ensures the 'offline' trained model can be loaded by `base.CustomGPRegression`.
    """
    from src.gp.base import CustomKernelFunctions
    
    # Create kernel using the adapted init we made in Phase 4
    kernel_func = CustomKernelFunctions('squared_exponential', params=params)
    
    # Compute K (Covariance Matrix)
    K = kernel_func(train_x, train_x) 
    # Add noise variance to diagonal
    K_noise = K + params.noise_variance * np.eye(len(train_x))
    
    # Invert and solve
    # Using cholesky for stability: K = L L^T
    try:
        L = np.linalg.cholesky(K_noise)
        # alpha = K^-1 (y - mean)
        # We model residuals y_res = y - mean using the Zero-Mean GP
        y_res = train_y - params.mean
        z = np.linalg.solve(L, y_res)
        k_inv_y = np.linalg.solve(L.T, z) # This is alpha
        
        # K_inv calculation (expensive but needed for CasADi implementation of variance)
        # K^-1 = (L^-1)^T (L^-1)
        L_inv = np.linalg.inv(L)
        k_inv = L_inv.T @ L_inv
        
    except np.linalg.LinAlgError:
         # Fallback to pseudo-inverse if unstable
        print("Warning: Cholesky failed, using pinv for GP matrices.")
        k_inv = np.linalg.pinv(K_noise)
        y_res = train_y - params.mean
        k_inv_y = k_inv @ y_res

    return {
        'k_inv': k_inv,
        'k_inv_y': k_inv_y,
        'x_train': train_x,
        'y_train': train_y,
        'kernel_type': 'squared_exponential',
        'kernel_params': {
            'l': np.array(params.length_scale),
            'sigma_f': np.sqrt(params.signal_variance)
        },
        'sigma_n': params.noise_variance,
        'sigma_n': params.noise_variance,
        # Default means (assuming 0 prior mean for simple GP)
        'mean': np.zeros(train_x.shape[1]), 
        'y_mean': params.mean
    }
