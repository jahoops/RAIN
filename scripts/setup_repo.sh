#!/bin/bash
# Initialize the repository structure with empty directories

# Create directory structure
mkdir -p data/{raw,processed,external}
mkdir -p models
mkdir -p notebooks
mkdir -p logs
mkdir -p rain/{core,utils,models,data}
mkdir -p tests/{unit,integration}

# Create empty .gitkeep files to track empty directories
touch data/.gitkeep
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/external/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep
touch notebooks/.gitkeep

# Make script executable
chmod +x "$(dirname "$0")/setup_repo.sh"

echo "Repository structure created successfully!"
