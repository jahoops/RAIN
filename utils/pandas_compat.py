"""
Compatibility utilities for pandas version differences
"""
import pandas as pd
import contextlib

@contextlib.contextmanager
def pandas_option_context_wrapper(*args, **kwargs):
    """
    A wrapper for pd.option_context that handles deprecated options
    
    This specifically works around the 'mode.use_inf_as_null' option which
    was removed in newer pandas versions.
    """
    # Filter out any options that don't exist in current pandas version
    valid_args = []
    for arg in args:
        if arg == 'mode.use_inf_as_null':
            # This option is deprecated, handle it manually
            continue
        valid_args.append(arg)
    
    # Apply the context with only valid options
    with pd.option_context(*valid_args, **kwargs) as context:
        yield context

def fix_pandas_compatibility():
    """
    Apply monkey patches to provide compatibility with older pandas API
    """
    # Save the original option_context
    original_option_context = pd.option_context
    
    # Replace with our wrapper
    pd.option_context = pandas_option_context_wrapper
    
    # Return a function to restore the original if needed
    def restore():
        pd.option_context = original_option_context
    
    return restore

def patch_seaborn():
    """
    Apply patch to make seaborn work with newer pandas versions
    """
    try:
        import seaborn as sns
        from seaborn._core import VectorPlotter
        
        # Save original method
        original_method = VectorPlotter.comp_data.__get__
        
        # Define a replacement method that avoids using the deprecated option
        def patched_comp_data(self):
            """Compute a data structure that can be indexed by semantic variables."""
            if hasattr(self, "_comp_data"):
                return self._comp_data
                
            # This is the fix - don't use the problematic option_context
            self._comp_data = self._compute_comp_data()
            return self._comp_data
        
        # Apply the monkey patch
        VectorPlotter.comp_data = property(patched_comp_data)
        print("Successfully patched seaborn for pandas compatibility")
    except Exception as e:
        print(f"Could not patch seaborn: {e}")

if __name__ == "__main__":
    # Apply the fixes
    restore_pandas = fix_pandas_compatibility()
    patch_seaborn()
    print("Applied pandas compatibility fixes")
