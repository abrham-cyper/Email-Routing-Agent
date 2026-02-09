import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
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
torch.set_num_threads(1)
os.makedirs("results", exist_ok=True)
device = "cpu"

# Load Data
print("Loading data...")
train_ds, val_ds, test_ds, label_list, label2id, id2label = load_and_prepare_data()

# Model
print("Loading DistilGPT-2 for LoRA...")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.to(device)

# Processing
def generate_prompt(text):
    return f"Classify the email into one of the following departments: Technical Support, Customer Service, Billing and Payments, Sales and Pre-Sales, General Inquiry.\n\nEmail: {text}\n\nDepartment:"

def format_ds(example):
    prompt = generate_prompt(example["body"])
    completion = f" {example['queue']}" + tokenizer.eos_token
    return {"text": prompt + completion}

print("Preparing dataset (subset)...")
# Tiny subset for CPU train run
train_subset = train_ds.select(range(50))
val_subset = val_ds.select(range(10))

train_data = train_subset.map(format_ds)
val_data = val_subset.map(format_ds)

def tokenize(examples):
    outputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_train = train_data.map(tokenize, batched=True)
tokenized_val = val_data.map(tokenize, batched=True)

# Train
args = TrainingArguments(
    output_dir="./results_agent2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    save_strategy="no",
    eval_strategy="no",
    use_cpu=True,
    dataloader_num_workers=0,
    logging_steps=5
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=tokenized_train, 
    eval_dataset=tokenized_val,
    data_collator=data_collator
)

print("Training Agent 2...")
trainer.train()

# Eval
print("Skipping Agent 2 Evaluation to prevent Bus Error on this machine...")
# model.eval()
# predictions = []
# eval_ds = test_ds.select(range(50))
# references = eval_ds["queue"]
# start_time = time.time()

# for item in tqdm(eval_ds):
#     prompt = generate_prompt(item["body"])
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id, do_sample=False)
#     gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     ans = gen[len(prompt):].strip().split('\n')[0]
#     
#     pred = "Unknown"
#     for l in label_list:
#         if l.lower() in ans.lower():
#             pred = l
#             break
#     predictions.append(pred)

# end_time = time.time()
# acc = accuracy_score(references, predictions)
# total_time = end_time - start_time

acc = 0.0
total_time = 0.0

print(f"Agent 2 Accuracy: {acc:.4f} (Skipped)")
print(f"Agent 2 Time: {total_time:.2f}s (Skipped)")

with open("results/agent2.json", "w") as f:
    json.dump({"accuracy": acc, "time": total_time}, f)
