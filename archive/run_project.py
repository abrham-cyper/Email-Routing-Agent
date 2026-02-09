import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datapreparation import load_and_prepare_data
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from peft import get_peft_model, LoraConfig, TaskType
import os
# Limit threads to avoid bus errors/crashes on resource-constrained environments
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Set device
# Set device - forcing CPU for stability on this machine due to Bus Error
device = "cpu"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Optional fallback
print(f"Using device: {device}")

# --- 1. Load Data ---
print("\n=== 1. Loading Data ===")
train_ds, val_ds, test_ds, label_list, label2id, id2label = load_and_prepare_data()
print(f"Train size: {len(train_ds)}")
print(f"Test size: {len(test_ds)}")

# --- Agent 1: GPT-2 Zero-Shot ---
print("\n=== 2. Running Agent 1: GPT-2 Zero-Shot ===")
model_name_gpt2 = "gpt2"
# Use a smaller model for faster CPU execution if needed, but sticking to gpt2 as requested
tokenizer_gpt2 = AutoTokenizer.from_pretrained(model_name_gpt2)
model_gpt2 = AutoModelForCausalLM.from_pretrained(model_name_gpt2).to(device)

if tokenizer_gpt2.pad_token is None:
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
    model_gpt2.config.pad_token_id = model_gpt2.config.eos_token_id

def generate_prompt_zeroshot(text):
    return f"Classify the email into one of the following departments: Technical Support, Customer Service, Billing and Payments, Sales and Pre-Sales, General Inquiry.\n\nEmail: {text}\n\nDepartment:"

def evaluate_gpt2_zeroshot(model, tokenizer, dataset, desc="Evaluating"):
    predictions = []
    references = dataset["queue"]
    start_time = time.time()
    
    # Use subset for quick testing if running on CPU
    if device == "cpu":
        print("Running on CPU - using smaller subset for speed in Agent 1 evaluation...")
        dataset = dataset.select(range(min(50, len(dataset))))
        references = dataset["queue"]
    
    for item in tqdm(dataset, desc=desc):
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

acc_agent1, time_agent1 = evaluate_gpt2_zeroshot(model_gpt2, tokenizer_gpt2, test_ds, desc="Agent 1 Eval")
print(f"Agent 1 Accuracy: {acc_agent1:.4f}")
print(f"Agent 1 Time: {time_agent1:.2f}s")

# --- Agent 2: GPT-2 LoRA ---
print("\n=== 3. Training Agent 2: GPT-2 LoRA ===")

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

def format_ds_agent2(example):
    prompt = generate_prompt_zeroshot(example["body"])
    completion = f" {example['queue']}" + tokenizer_gpt2.eos_token
    text = prompt + completion
    return {"text": text}

# Limit training data for speed if on CPU
train_data_subset = train_ds
val_data_subset = val_ds
if device == "cpu":
    print("Running on CPU - using smaller training subset for Agent 2...")
    train_data_subset = train_ds.select(range(min(100, len(train_ds))))
    val_data_subset = val_ds.select(range(min(20, len(val_ds))))

train_ds_lora = train_data_subset.map(format_ds_agent2)
val_ds_lora = val_data_subset.map(format_ds_agent2)

def tokenize_function_causal(examples):
    return tokenizer_gpt2(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train_lora = train_ds_lora.map(tokenize_function_causal, batched=True)
tokenized_val_lora = val_ds_lora.map(tokenize_function_causal, batched=True)

training_args_lora = TrainingArguments(
    output_dir="./results_agent2",
    per_device_train_batch_size=1, # Reduced to 1 for stability
    gradient_accumulation_steps=4, # Simulate larger batch
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no", # Save space/time
    evaluation_strategy="no",
    use_cpu=True,
    dataloader_num_workers=0 # reduced to 0 to avoid multiprocessing issues
)

trainer_lora = Trainer(
    model=model_gpt2_lora,
    args=training_args_lora,
    train_dataset=tokenized_train_lora,
    eval_dataset=tokenized_val_lora,
)

trainer_lora.train()

# Eval Agent 2
print("Evaluating Agent 2...")
model_gpt2_lora.eval()
acc_agent2, time_agent2 = evaluate_gpt2_zeroshot(model_gpt2_lora, tokenizer_gpt2, test_ds, desc="Agent 2 Eval")
print(f"Agent 2 Accuracy: {acc_agent2:.4f}")
print(f"Agent 2 Time: {time_agent2:.2f}s")


# --- Agent 3: DistilBERT Classifier ---
print("\n=== 4. Training Agent 3: DistilBERT ===")

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

# Limit data if CPU
train_bert_subset = train_ds
val_bert_subset = val_ds
test_bert_subset = test_ds
if device == "cpu":
    print("Running on CPU - using smaller training subset for Agent 3...")
    train_bert_subset = train_ds.select(range(min(100, len(train_ds))))
    val_bert_subset = val_ds.select(range(min(20, len(val_ds))))
    test_bert_subset = test_ds.select(range(min(50, len(test_ds))))

tokenized_train_bert = train_bert_subset.map(tokenize_function_bert, batched=True)
tokenized_val_bert = val_bert_subset.map(tokenize_function_bert, batched=True)
tokenized_test_bert = test_bert_subset.map(tokenize_function_bert, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer_bert)

training_args_bert = TrainingArguments(
    output_dir="./results_agent3",
    learning_rate=2e-5,
    per_device_train_batch_size=4, # Reduced for stability
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1, # Reduced to 1 for speed demonstration, user can increase
    weight_decay=0.01,
    evaluation_strategy="no",
    save_strategy="no",
    use_cpu=True,
    dataloader_num_workers=0
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

trainer_bert.train()

# Eval Agent 3
print("Evaluating Agent 3...")
start_time = time.time()
metrics = trainer_bert.evaluate(tokenized_test_bert)
end_time = time.time()

acc_agent3 = metrics["eval_accuracy"]
time_agent3 = end_time - start_time
print(f"Agent 3 Accuracy: {acc_agent3:.4f}")
print(f"Agent 3 Time: {time_agent3:.2f}s")


# --- Comparison ---
print("\n=== 5. Comparison ===")
methods = ['GPT-2 Zero-Shot', 'GPT-2 LoRA', 'DistilBERT']
accuracies = [acc_agent1, acc_agent2, acc_agent3]
times = [time_agent1, time_agent2, time_agent3]

print(f"{'Method':<20} | {'Accuracy':<10} | {'Time (s)':<10}")
print("-" * 44)
for m, a, t in zip(methods, accuracies, times):
    print(f"{m:<20} | {a:<10.4f} | {t:<10.2f}")

# Plot
try:
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
    plt.savefig("comparison_chart.png") # Save instead of show
    print("\nChart saved as 'comparison_chart.png'")
except Exception as e:
    print(f"Could not save plot: {e}")

print("\nDONE! You can see the results above.")
