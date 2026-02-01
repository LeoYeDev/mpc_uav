import sys
import os
import numpy as np
import pandas as pd
import shutil
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_offline_training_flow():
    print("Testing Phase 5: Offline Training Unification...")
    
    # 1. Setup Mock Data
    # We need to mock read_dataset or GPDataset to provide simple data
    from src.gp.utils import GPDataset
    
    # Create fake data
    N = 50
    x_raw = np.random.randn(N, 13) # State
    u_raw = np.random.randn(N, 4)  # Input
    y_raw = np.sin(x_raw) * 0.1    # Fake Model Error
    dt = 0.01 * np.ones(N)
    
    # Init GPDataset with mock data directly (bypassing load_data)
    # We subclass or just inject
    gp_ds = GPDataset()
    gp_ds.x_raw = x_raw
    gp_ds.u_raw = u_raw
    gp_ds.y_raw = y_raw
    gp_ds.x_out_raw = x_raw + y_raw * dt[:, np.newaxis] # Approx
    gp_ds.x_pred_raw = x_raw
    gp_ds.dt_raw = dt
    gp_ds.pruned_idx = list(range(N))
    gp_ds.cluster_agency = {0: list(range(N))}
    gp_ds.centroids = np.zeros((1, 13))
    gp_ds.x_features = [7, 8, 9] 
    gp_ds.u_features = []
    gp_ds.y_dim = 7 # Predict velocity x error
    
    # 2. Run Main Training Logic (partial)
    # We import the internal function or a wrapped main
    from src.gp.offline import gp_train_and_save_torch
    
    save_path = "tests/temp_models"
    save_file = "test_model"
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    try:
        # Get data chunks similar to what 'main' does
        x_train = x_raw[:, [7, 8, 9]]
        y_train = y_raw[:, 7]
        
        print("Running PyTorch training...")
        results = gp_train_and_save_torch(
            x=[x_train], 
            y=[y_train], 
            save_model=True, 
            save_file=save_file, 
            save_path=save_path, 
            y_dims=[7], 
            cluster_n=0,
            progress_bar=False
        )
        
        print("Training finished.")
        assert len(results) == 1
        model_dict = results[0]
        
        # 3. Verify Output Structure
        required_keys = ['k_inv', 'k_inv_y', 'kernel_params', 'sigma_n']
        for k in required_keys:
            assert k in model_dict, f"Missing key {k} in saved model dict"
            
        print("Keys verification passed.")
        print(f"Learned LengthScale: {model_dict['kernel_params']['l']}")
        print(f"Learned SigmaN: {model_dict['sigma_n']}")
        
        # 4. Verify Loading into CasADi Regressor
        from src.gp.base import CustomGPRegression
        gp_reg = CustomGPRegression(x_features=[7,8,9], u_features=[], reg_dim=7)
        gp_reg.load(model_dict)
        
        # Test valid prediction
        # Input should be (N, D). Our features are [7, 8, 9] so D=3.
        # CustomGPRegression.predict returns the mean array directly if return_std=False
        res = gp_reg.predict(np.zeros((1, 3)), np.zeros((1, 0)))
        print(f"Prediction result shape: {res.shape}")
        assert res.shape == (1, 1) or res.shape == (1,)
        
        print("CasADi loading and prediction passed.")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

if __name__ == "__main__":
    test_offline_training_flow()
