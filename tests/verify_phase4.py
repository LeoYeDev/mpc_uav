import sys
import os
import torch
import numpy as np
from typing import List

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_unified_gp_interface():
    print("Testing Unified GP Interface...")
    try:
        from config.gp_config import GPModelParams, OnlineGPConfig
        from src.gp.base import CustomKernelFunctions
        from src.gp.online import IncrementalGP
        
        # 1. Test GPModelParams creation
        params = GPModelParams(
            length_scale=[1.5],
            signal_variance=2.0,
            noise_variance=0.1
        )
        print("GPModelParams created.")
        
        # 2. Test base.py adapter
        kernel = CustomKernelFunctions(kernel_func='squared_exponential', params=params)
        print("CustomKernelFunctions adapter worked.")
        print(f"Internal params: {kernel.params}")
        assert kernel.params['l'] == [1.5]
        # CustomKernelFunctions stores sigma_f (amplitude), so it should be sqrt(variance)
        assert np.isclose(kernel.params['sigma_f'], np.sqrt(2.0))
        
        # 3. Test online.py parameter export
        # Mocking an IncrementalGP
        config = OnlineGPConfig().to_dict()
        gp = IncrementalGP(dim_idx=0, config=config)
        
        # Set some arbitrary values to the internal model
        new_lengthscale = torch.tensor([[0.8]])
        new_outputscale = torch.tensor(1.2)
        new_noise = torch.tensor(0.05)
        
        # GPyTorch handling is a bit complex with constraints, but we can try setting direct attributes
        # Note: In a real scenario we'd use state_dict or training, here we stick to verifying the export structure
        # works if values exist.
        
        exported_params = gp.get_model_params()
        print("Exported params from IncrementalGP:")
        print(exported_params)
        assert isinstance(exported_params, GPModelParams)
        assert isinstance(exported_params.length_scale, list)
        
        print("Unified GP Interface verification successful.")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_unified_gp_interface()
