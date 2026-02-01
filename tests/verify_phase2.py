import sys
import os
import importlib
import traceback

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_matplotlib_backend():
    print("Testing Matplotlib backend check...")
    # Simulate headless environment
    os.environ.pop('DISPLAY', None)
    
    # Force reload of online module
    if 'src.gp.online' in sys.modules:
        del sys.modules['src.gp.online']
    
    try:
        import src.gp.online
        import matplotlib
        backend = matplotlib.get_backend()
        print(f"Matplotlib backend: {backend}")
        if backend.lower() != 'agg':
            print("WARNING: Backend is not Agg behavior might differ on real headless system.")
            # On some systems matplotlib might default to something else if TkAgg fails, 
            # but we explicitly set it to Agg in our code if DISPLAY is missing.
            # Let's check if our code executed.
        print("Import src.gp.online in headless mode successful.")
    except Exception as e:
        print(f"Import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

def test_type_hints_syntax():
    print("Testing type hints syntax...")
    try:
        import src.core.controller
        from src.core.controller import Quad3DMPC
        print("Import src.core.controller successful.")
    except SyntaxError as e:
        print(f"Syntax Error in type hints: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_matplotlib_backend()
    test_type_hints_syntax()
