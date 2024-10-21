from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
import sys

sys.stdout.reconfigure(encoding='utf-8')  # Ensure encoding for console output

# Load the pretrained LLaMA 3.2-1B model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Ensure that pad_token is set

# Define LoRA configuration for fine-tuning
lora_config = LoraConfig(
    r=8,               # Low-rank matrix dimension
    lora_alpha=32,     # LoRA alpha hyperparameter
    target_modules=["q_proj", "v_proj"],  # Apply to self-attention layers
    lora_dropout=0.05, # Dropout to prevent overfitting
    bias="none"        # Don't use bias
)

# Apply LoRA to the LLaMA model
lora_model = get_peft_model(model, lora_config)

# Load the question-answer dataset
def load_qa_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        qa_data = f.read().split("--------------------------------------------------------------------------------")
    qa_pairs = []
    for qa in qa_data:
        lines = qa.strip().split("\n")
        if len(lines) >= 2:
            question = lines[1].replace("Question:", "").strip()
            answer = lines[2].replace("Answer:", "").strip()
            qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

# Load and prepare the dataset for training
qa_pairs = load_qa_data("prompts_without_context_output.txt")
qa_dataset = Dataset.from_dict({"question": [pair["question"] for pair in qa_pairs], "answer": [pair["answer"] for pair in qa_pairs]})

# Define data preprocessing function with an explicit instruction format
def preprocess_function(examples):
    # Add instruction to the input: "Answer the following question: "
    inputs = tokenizer(["Question: " + question + " Answer:" for question in examples["question"]],
                       padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = tokenizer(examples["answer"], padding="max_length", truncation=True, max_length=128)["input_ids"]
    return inputs

# Tokenize the dataset
tokenized_dataset = qa_dataset.map(preprocess_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./lora_llama_finetune",  # Output directory for checkpoints
    per_device_train_batch_size=2,       # Batch size per device for training
    per_device_eval_batch_size=2,        # Batch size for evaluation
    num_train_epochs=2,                  # Number of training epochs
    logging_dir="./logs",                # Log directory
    learning_rate=5e-5,                  # Learning rate for fine-tuning
)

# Initialize the Trainer with the LoRA model and the fine-tuning data
trainer = Trainer(
    model=lora_model,                    # The LoRA-finetuned model
    args=training_args,                  # Training arguments
    train_dataset=tokenized_dataset,     # Training dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
lora_model.save_pretrained("./lora_llama_finetuned")
