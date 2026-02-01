import numpy as np
import torch
import gpytorch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.gp.train_utils import train_gp_torch, extract_gp_params, compute_matrices_for_casadi
from src.gp.base import CustomGPRegression

def test_equivalence():
    print("Testing PyTorch vs CasADi equivalence...")
    
    # 1. Generate Data with non-zero mean
    np.random.seed(42)
    # y = sin(x) + 2.0 (mean offset)
    X = np.linspace(0, 5, 20).reshape(-1, 1)
    Y = np.sin(X).flatten() + 2.0
    
    # 2. Train PyTorch Model
    print("Training PyTorch model...")
    model, likelihood = train_gp_torch(X, Y, n_iter=100, lr=0.1, verbose=False)
    
    # Check if mean module learned something close to 2.0
    learned_mean = model.mean_module.constant.item()
    print(f"Learned Mean: {learned_mean:.4f} (Expected ~2.0)")
    
    # 3. Export to CasADi format
    gp_params = extract_gp_params(model, likelihood)
    print(f"Exported Mean: {gp_params.mean:.4f}")
    
    model_dict = compute_matrices_for_casadi(gp_params, X, Y)
    
    # 4. Load into CasADi Regressor
    # reg_dim=0, x_features=[0]
    casadi_gp = CustomGPRegression(x_features=[0], u_features=[], reg_dim=0)
    casadi_gp.load(model_dict)
    
    # 5. Compare Predictions
    test_X_np = np.linspace(0, 5, 5).reshape(-1, 1)
    test_X_torch = torch.from_numpy(test_X_np).float()
    
    # PyTorch Pred
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        torch_pred = likelihood(model(test_X_torch)).mean.numpy()
        
    # CasADi Pred
    # predict returns (N,) array
    casadi_pred = casadi_gp.predict(test_X_np)
    
    # Error
    err = np.max(np.abs(torch_pred - casadi_pred))
    print(f"Max Prediction Error: {err:.6e}")
    
    if err < 1e-3:
        print("PASS: Predictions match.")
    else:
        print("FAIL: Predictions mismatch.")
        print(f"Torch: {torch_pred}")
        print(f"CasADi: {casadi_pred}")
        sys.exit(1)

if __name__ == "__main__":
    test_equivalence()
