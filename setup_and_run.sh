#!/bin/bash

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run verification
echo "Running data verification..."
python verify_data.py

if [ "$1" == "--script" ]; then
    # Run the python script directly
    echo "Running project script..."
    python run_project.py
else
    # Launch Jupyter Notebook
    echo "Launching Jupyter Notebook..."
    echo "Please use the link above to open the notebook in your browser."
    jupyter notebook project.ipynb
fi
