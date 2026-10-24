#!/usr/bin/env python3

import csv
import os
import re
from collections import defaultdict

# Parse the CSV file
with open("/home/USER/Projects/RAIN/submission_results.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Group by score
score_groups = defaultdict(list)
for row in rows:
    score = float(row["score"])
    if score > 0.85:  # Only include scores > 0.85
        score_groups[score].append(row["path"])

# Create destination directory
os.makedirs("/home/USER/Projects/RAIN//data/candidates", exist_ok=True)

# Function to normalize filenames to identify duplicates
def normalize_filename(path):
    """Extract base filename without variations"""
    filename = os.path.basename(path)
    # Remove _fixed, _fixed3, etc. suffixes
    return re.sub(r'_fixed[0-9]*|_fixed_fixed[0-9]*', '', filename)

# Process each score group
copied_files = []
for score in sorted(score_groups.keys(), reverse=True):
    paths = score_groups[score]
    
    # Group by normalized filename
    unique_files = {}
    for path in paths:
        norm_name = normalize_filename(path)
        if norm_name not in unique_files:
            unique_files[norm_name] = []
        unique_files[norm_name].append(path)
    
    # For each unique file, select one representative
    for norm_name, file_paths in unique_files.items():
        # Prefer files from Actual directory if available
        actual_paths = [p for p in file_paths if "/Actual/" in p]
        if actual_paths:
            selected = actual_paths[0]
        else:
            # Otherwise choose the shortest path
            selected = min(file_paths, key=len)
        
        dest_file = f"/home/USER/Projects/RAIN/data/candidates/{os.path.basename(selected)}"
        os.system(f"cp '{selected}' '{dest_file}'")
        copied_files.append((selected, score, dest_file))

print(f"Copied {len(copied_files)} unique high-scoring files to /data/candidates")
print("\nHighest scoring submissions:")
for path, score, dest in copied_files[:5]:
    print(f"- {path} (score: {score})")