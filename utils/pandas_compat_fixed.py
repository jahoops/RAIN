"""
Fixed compatibility utilities for pandas version differences
"""
import pandas as pd
import contextlib
import warnings

# Save reference to original function to avoid recursion
_original_option_context = pd.option_context

@contextlib.contextmanager
def pandas_option_context_wrapper(*args, **kwargs):
    """
    A wrapper for pd.option_context that handles deprecated options
    without causing recursion.
    """
    # Filter out the problematic option
    valid_args = []
    for arg in args:
        if arg == 'mode.use_inf_as_null':
            warnings.warn("'mode.use_inf_as_null' is deprecated and will be ignored")
            continue
        valid_args.append(arg)
    
    # Use the original function to avoid recursion
    if valid_args:  # Only call if we have valid arguments
        with _original_option_context(*valid_args, **kwargs) as context:
            yield context
    else:
        # If no valid args, just use a dummy context manager
        @contextlib.contextmanager
        def dummy_context():
            yield None
        
        with dummy_context() as context:
            yield context

def fix_pandas_compatibility():
    """
    Apply monkey patch to provide compatibility with older pandas API
    """
    # Replace with our wrapper - only if it hasn't been replaced already
    if pd.option_context is _original_option_context:
        pd.option_context = pandas_option_context_wrapper
        print("Applied pandas option_context patch")
    
    # Return a function to restore the original if needed
    def restore():
        pd.option_context = _original_option_context
    
    return restore

def monkey_patch_seaborn():
    """
    Apply a monkeypatch to make seaborn work with newer pandas versions
    without using the deprecated option_context
    """
    try:
        import seaborn as sns
        from seaborn._core import VectorPlotter
        
        # Define a replacement method that avoids using the problematic option
        original_comp_data = VectorPlotter.comp_data
        
        @property
        def patched_comp_data(self):
            """Compute a data structure that can be indexed by semantic variables."""
            if hasattr(self, "_comp_data"):
                return self._comp_data
            
            # This is the fix - don't use problematic option_context
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._comp_data = self._compute_comp_data()
            
            return self._comp_data
        
        # Apply the patch
        VectorPlotter.comp_data = patched_comp_data
        print("Successfully patched seaborn for pandas compatibility")
        
        # Return function to restore original method if needed
        def restore():
            VectorPlotter.comp_data = original_comp_data
        
        return restore
    except Exception as e:
        print(f"Could not patch seaborn: {e}")
        return lambda: None  # Return dummy restore function

if __name__ == "__main__":
    # Apply the fixes
    restore_pandas = fix_pandas_compatibility()
    restore_seaborn = monkey_patch_seaborn()
    print("Applied pandas compatibility fixes")
    
    # Test the patch
    try:
        with pd.option_context('display.max_rows', 10):
            print("pd.option_context works correctly")
    except Exception as e:
        print(f"Error using patched option_context: {e}")
