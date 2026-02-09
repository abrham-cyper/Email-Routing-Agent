# 📧 AI Email Routing Agent (NLP 2025-26)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-DistilBERT%20%7C%20GPT2-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Objective
Develop an intelligent system to route customer emails to the correct department:
1.  **Technical Support**
2.  **Customer Service**
3.  **Billing and Payments**
4.  **Sales and Pre-Sales**
5.  **General Inquiry**

## 🧠 Solved using 3 AI Agents:
*   **Agent 1**: Zero-shot prompting with **DistilGPT-2**.
*   **Agent 2**: Efficient Fine-tuning (PEFT/LoRA) with **DistilGPT-2**.
*   **Agent 3**: Sequence Classification with **DistilBERT** (Best Performer).

## 🚀 How to Run

### Option A: Google Colab (Recommended)
1.  Upload `project.ipynb`.
2.  Upload `src.zip`.
3.  Run the first cell to unzip: `!unzip src.zip`.
4.  Run all cells.

### Option B: Local Machine
```bash
# Setup and Run
./run.sh
```

## 📂 Project Structure
*   `src/`: Core logic and agent implementations.
*   `project.ipynb`: Main notebook for results and presentation.
*   `results/`: Generated metrics and comparison charts.

