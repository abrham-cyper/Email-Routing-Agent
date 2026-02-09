#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install -r requirements.txt

echo "=========================================="
echo "RUNNING AGENT 1 (GPT-2 Zero Shot)"
echo "=========================================="
echo "Skipping Agent 1 due to persistent Bus Error on this machine."
# python src/agent1.py

echo ""
echo "=========================================="
echo "RUNNING AGENT 2 (GPT-2 LoRA)"
echo "=========================================="
python src/agent2.py

echo ""
echo "=========================================="
echo "RUNNING AGENT 3 (DistilBERT)"
echo "=========================================="
python src/agent3.py

echo ""
echo "=========================================="
echo "GENERATING RESULTS"
echo "=========================================="
python src/visualize.py

echo ""
echo "Cleaning up temporary training folders..."
rm -rf results_agent2 results_agent3

echo ""
echo "DONE! Results are saved in 'results/' folder."
