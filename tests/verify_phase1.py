import sys
import os
import time
import signal
from multiprocessing import Process

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_config_import():
    print("Testing config import...")
    try:
        from config.paths import ACADOS_MODELS_DIR, DEFAULT_MODEL_VERSION
        print(f"ACADOS_MODELS_DIR: {ACADOS_MODELS_DIR}")
        assert os.path.exists(os.path.dirname(ACADOS_MODELS_DIR))
        print("Config import successful.")
    except Exception as e:
        print(f"Config import failed: {e}")
        sys.exit(1)

def run_manager():
    from src.gp.online import IncrementalGPManager
    from config.gp_config import OnlineGPConfig
    
    # Run a manager for a few seconds then exit
    config = OnlineGPConfig().to_dict()
    config['num_dimensions'] = 1 # Minimize resource usage
    manager = IncrementalGPManager(config)
    print("Manager started. Sleeping...")
    time.sleep(5)
    print("Manager sleeping done.")

def test_cleanup():
    print("Testing cleanup with SIGTERM...")
    p = Process(target=run_manager)
    p.start()
    pid = p.pid
    print(f"Process started with PID {pid}")
    time.sleep(2)
    print("Sending SIGTERM...")
    os.kill(pid, signal.SIGTERM)
    p.join(timeout=5)
    
    if p.is_alive():
        print("Process failed to terminate!")
        p.terminate()
        sys.exit(1)
    else:
        print(f"Process {pid} terminated successfully. Exit code: {p.exitcode}")

if __name__ == "__main__":
    test_config_import()
    test_cleanup()
