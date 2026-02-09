import json

# Helper to split source into lines with newlines
def to_source(text):
    lines = text.split('\n')
    # Add \n to each line except the last one if it's empty check?
    # Actually, standard way is: [line + "\n" for line in lines[:-1]] + [lines[-1]] if needed
    # But simpler: just ensure every internal newline is preserved.
    # .split('\n') eats the newline.
    # We want to reconstruct it.
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]

# --- Setup & Data ---
source_setup = to_source(r"""
!pip install transformers datasets torch accelerate peft scikit-learn tqdm matplotlib --quiet
""")

source_data = to_source(r"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
# from config import * # Not needed
from datapreparation import load_and_prepare_data
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
from peft import get_peft_model, LoraConfig, TaskType

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Data
train_ds, val_ds, test_ds, label_list, label2id, id2label = load_and_prepare_data()

print(f"Train size: {len(train_ds)}")
print(f"Test size: {len(test_ds)}")
""")


# --- Agent 1: GPT-2 Zero-Shot ---
source_agent1 = to_source(r"""
# --- Agent 1: GPT-2 Prompting (Zero-shot) ---

model_name_gpt2 = "gpt2"
tokenizer_gpt2 = AutoTokenizer.from_pretrained(model_name_gpt2)
model_gpt2 = AutoModelForCausalLM.from_pretrained(model_name_gpt2).to(device)

if tokenizer_gpt2.pad_token is None:
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
    model_gpt2.config.pad_token_id = model_gpt2.config.eos_token_id

def generate_prompt_zeroshot(text):
    return f"Classify the email into one of the following departments: Technical Support, Customer Service, Billing and Payments, Sales and Pre-Sales, General Inquiry.\n\nEmail: {text}\n\nDepartment:"

def evaluate_gpt2_zeroshot(model, tokenizer, dataset):
    predictions = []
    references = dataset["queue"]
    start_time = time.time()
    
    # Use subset for quick testing if needed
    # dataset = dataset.select(range(100))
    
    for item in tqdm(dataset, desc="Evaluating Agent 1"):
        prompt = generate_prompt_zeroshot(item["body"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10, 
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract answer
        answer = generated_text[len(prompt):].strip().split('\n')[0]
        
        # Simple mapping
        pred_label = "Unknown"
        for label in label_list:
            if label.lower() in answer.lower():
                pred_label = label
                break
        predictions.append(pred_label)
        
    end_time = time.time()
    accuracy = accuracy_score(references, predictions)
    return accuracy, end_time - start_time

acc_agent1, time_agent1 = evaluate_gpt2_zeroshot(model_gpt2, tokenizer_gpt2, test_ds)
print(f"Agent 1 Accuracy: {acc_agent1:.4f}")
print(f"Agent 1 Time: {time_agent1:.2f}s")
""")

# --- Agent 2: GPT-2 LoRA ---
source_agent2_train = to_source(r"""
# --- Agent 2: GPT-2 Fine-tuning (LoRA) ---

# PEFT Config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

model_gpt2_lora = AutoModelForCausalLM.from_pretrained(model_name_gpt2)
model_gpt2_lora = get_peft_model(model_gpt2_lora, peft_config)
model_gpt2_lora.print_trainable_parameters()
model_gpt2_lora.to(device)

# Prepare Data for Causal LM training
def format_ds_agent2(example):
    prompt = generate_prompt_zeroshot(example["body"])
    completion = f" {example['queue']}" + tokenizer_gpt2.eos_token
    text = prompt + completion
    return {"text": text}

train_ds_lora = train_ds.map(format_ds_agent2)
val_ds_lora = val_ds.map(format_ds_agent2)

def tokenize_function_causal(examples):
    return tokenizer_gpt2(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train_lora = train_ds_lora.map(tokenize_function_causal, batched=True)
tokenized_val_lora = val_ds_lora.map(tokenize_function_causal, batched=True)

# Training
training_args_lora = TrainingArguments(
    output_dir="./results_agent2",
    per_device_train_batch_size=4,
    num_train_epochs=1, # Increase for better results
    learning_rate=2e-4,
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch", # or "no" if val set is small
    use_cpu=not torch.cuda.is_available()
)

trainer_lora = Trainer(
    model=model_gpt2_lora,
    args=training_args_lora,
    train_dataset=tokenized_train_lora,
    eval_dataset=tokenized_val_lora,
)

print("Training Agent 2...")
trainer_lora.train()
""")

source_agent2_eval = to_source(r"""
# Evaluate Agent 2
model_gpt2_lora.eval()
acc_agent2, time_agent2 = evaluate_gpt2_zeroshot(model_gpt2_lora, tokenizer_gpt2, test_ds)
print(f"Agent 2 Accuracy: {acc_agent2:.4f}")
print(f"Agent 2 Time: {time_agent2:.2f}s")
""")


# --- Agent 3: DistilBERT Classifier ---
source_agent3_train = to_source(r"""
# --- Agent 3: DistilBERT Classifier ---

model_name_bert = "distilbert-base-uncased"
tokenizer_bert = AutoTokenizer.from_pretrained(model_name_bert)
model_bert = DistilBertForSequenceClassification.from_pretrained(
    model_name_bert, 
    num_labels=len(label_list), 
    id2label=id2label, 
    label2id=label2id
).to(device)

def tokenize_function_bert(examples):
    return tokenizer_bert(examples["body"], truncation=True, max_length=512)

tokenized_train_bert = train_ds.map(tokenize_function_bert, batched=True)
tokenized_val_bert = val_ds.map(tokenize_function_bert, batched=True)
tokenized_test_bert = test_ds.map(tokenize_function_bert, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer_bert)

training_args_bert = TrainingArguments(
    output_dir="./results_agent3",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_cpu=not torch.cuda.is_available()
)

trainer_bert = Trainer(
    model=model_bert,
    args=training_args_bert,
    train_dataset=tokenized_train_bert,
    eval_dataset=tokenized_val_bert,
    tokenizer=tokenizer_bert,
    data_collator=data_collator,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}
)

print("Training Agent 3...")
trainer_bert.train()
""")

source_agent3_eval = to_source(r"""
# Evaluate Agent 3
start_time = time.time()
metrics = trainer_bert.evaluate(tokenized_test_bert)
end_time = time.time()

acc_agent3 = metrics["eval_accuracy"]
time_agent3 = end_time - start_time

print(f"Agent 3 Accuracy: {acc_agent3:.4f}")
print(f"Agent 3 Time: {time_agent3:.2f}s")
""")


# --- Comparison ---
source_comparison = to_source(r"""
# --- Comparison ---
methods = ['GPT-2 Zero-Shot', 'GPT-2 LoRA', 'DistilBERT']
accuracies = [acc_agent1, acc_agent2, acc_agent3]
times = [time_agent1, time_agent2, time_agent3]

print(f"{'Method':<20} | {'Accuracy':<10} | {'Time (s)':<10}")
print("-" * 44)
for m, a, t in zip(methods, accuracies, times):
    print(f"{m:<20} | {a:<10.4f} | {t:<10.2f}")

# Plot
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Method')
ax1.set_ylabel('Accuracy', color=color)
ax1.bar(methods, accuracies, color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Time (s)', color=color)  
ax2.plot(methods, times, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title("Method Comparison")
plt.show()
""")


# Create the notebook structure
cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# NLP Project 2025: LLM Email Routing Agents"]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Setup & Data Loading"]},
    {"cell_type": "code", "source": source_setup, "execution_count": None, "outputs": [], "metadata": {}},
    {"cell_type": "code", "source": source_data, "execution_count": None, "outputs": [], "metadata": {}},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Agent 1: GPT-2 Prompting (Zero-shot)"]},
    {"cell_type": "code", "source": source_agent1, "execution_count": None, "outputs": [], "metadata": {}},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Agent 2: GPT-2 Fine-tuning (LoRA)"]},
    {"cell_type": "code", "source": source_agent2_train, "execution_count": None, "outputs": [], "metadata": {}},
    {"cell_type": "code", "source": source_agent2_eval, "execution_count": None, "outputs": [], "metadata": {}},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Agent 3: DistilBERT Classifier"]},
    {"cell_type": "code", "source": source_agent3_train, "execution_count": None, "outputs": [], "metadata": {}},
    {"cell_type": "code", "source": source_agent3_eval, "execution_count": None, "outputs": [], "metadata": {}},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Comparison & Results"]},
    {"cell_type": "code", "source": source_comparison, "execution_count": None, "outputs": [], "metadata": {}}
]

nb = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('project.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Created project.ipynb")
