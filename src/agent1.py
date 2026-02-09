import torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from datapreparation import load_and_prepare_data
except ImportError:
    from src.datapreparation import load_and_prepare_data
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time
import os
import json

# Setup
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
os.makedirs("results", exist_ok=True)
device = "cpu"
print(f"Using device: {device}")

# Load Data
print("Loading data...")
train_ds, val_ds, test_ds, label_list, label2id, id2label = load_and_prepare_data()

# Model
print("Loading DistilGPT-2 (Smaller/Faster)...")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def generate_prompt(text):
    return f"Classify the email into one of the following departments: Technical Support, Customer Service, Billing and Payments, Sales and Pre-Sales, General Inquiry.\n\nEmail: {text}\n\nDepartment:"

# Eval
print("Evaluating Agent 1 (Subset for speed on CPU)...")
predictions = []
# Use small subset for CPU testing
eval_ds = test_ds.select(range(50)) 
references = eval_ds["queue"]
start_time = time.time()

for item in tqdm(eval_ds):
    prompt = generate_prompt(item["body"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = gen_text[len(prompt):].strip().split('\n')[0]
    
    pred_label = "Unknown"
    for label in label_list:
        if label.lower() in answer.lower():
            pred_label = label
            break
    predictions.append(pred_label)

end_time = time.time()
acc = accuracy_score(references, predictions)
total_time = end_time - start_time

print(f"Agent 1 Accuracy: {acc:.4f}")
print(f"Agent 1 Time: {total_time:.2f}s")

# Save results
with open("results/agent1.json", "w") as f:
    json.dump({"accuracy": acc, "time": total_time}, f)
