import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
try:
    from datapreparation import load_and_prepare_data
except ImportError:
    from src.datapreparation import load_and_prepare_data
import numpy as np
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
print("Loading DistilBERT...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
).to(device)

def tokenize(examples):
    return tokenizer(examples["body"], truncation=True, max_length=512)

def map_labels(example):
    return {"label": label2id[example["queue"]]}

# SMART DEVICE DETECTION
if torch.cuda.is_available():
    print("🚀 GPU Detected! Usage: MEDIUM SUBSET (2000 samples to save time)")
    device = "cuda"
    train_subset = train_ds.select(range(2000)).map(map_labels)
    val_subset = val_ds.select(range(200)).map(map_labels) 
    test_subset = test_ds.select(range(500)).map(map_labels)
    use_cpu_flag = False
    batch_size = 16
    epochs = 3
else:
    print("⚠️  CPU Detected! Usage: Tiny Subset (50 samples)")
    device = "cpu"
    train_subset = train_ds.select(range(50)).map(map_labels)
    val_subset = val_ds.select(range(10)).map(map_labels)
    test_subset = test_ds.select(range(50)).map(map_labels)
    use_cpu_flag = True
    batch_size = 4
    epochs = 1

model.to(device)

tokenized_train = train_subset.map(tokenize, batched=True)
tokenized_val = val_subset.map(tokenize, batched=True)
tokenized_test = test_subset.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir="./results_agent3",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size, 
    num_train_epochs=epochs,
    use_cpu=use_cpu_flag,
    dataloader_num_workers=0,
    logging_steps=5,
    eval_strategy="no",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}
)

print("Training Agent 3...")
trainer.train()

print("Evaluating Agent 3...")
start_time = time.time()
metrics = trainer.evaluate(tokenized_test)
end_time = time.time()

acc = metrics["eval_accuracy"]
total_time = end_time - start_time

print(f"Agent 3 Accuracy: {acc:.4f}")
print(f"Agent 3 Time: {total_time:.2f}s")

with open("results/agent3.json", "w") as f:
    json.dump({"accuracy": acc, "time": total_time}, f)
