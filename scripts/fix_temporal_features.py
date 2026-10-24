import pandas as pd

def fix_inplace_warnings():
    """Fix pandas FutureWarning by examining and correcting temporal_features.py"""
    
    # Read the current file
    with open('temporal_features.py', 'r') as file:
        lines = file.readlines()
    
    # Find and fix problematic lines
    fixed_lines = []
    for i, line in enumerate(lines):
        if '.fillna(' in line and 'inplace=True' in line:
            # Extract the DataFrame and column
            parts = line.split('.fillna(')
            df_col = parts[0].strip()
            params = parts[1].strip()
            
            # Create the fixed version
            indent = len(line) - len(line.lstrip())
            fixed_line = ' ' * indent + f"{df_col} = {df_col}.fillna({params.replace('inplace=True', '').replace(', )', ')')}\n"
            fixed_lines.append(fixed_line)
            print(f"Line {i+1}: Fixed fillna inplace operation")
        elif '.replace(' in line and 'inplace=True' in line:
            # Extract the DataFrame and column
            parts = line.split('.replace(')
            df_col = parts[0].strip()
            params = parts[1].strip()
            
            # Create the fixed version
            indent = len(line) - len(line.lstrip())
            fixed_line = ' ' * indent + f"{df_col} = {df_col}.replace({params.replace('inplace=True', '').replace(', )', ')')}\n"
            fixed_lines.append(fixed_line)
            print(f"Line {i+1}: Fixed replace inplace operation")
        else:
            fixed_lines.append(line)
    
    # Write the fixed file
    with open('temporal_features_fixed.py', 'w') as file:
        file.writelines(fixed_lines)
    
    print("\nCreated fixed version: temporal_features_fixed.py")
    print("Please review and update your imports to use the fixed version.")

if __name__ == "__main__":
    fix_inplace_warnings()