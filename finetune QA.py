from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
import sys

sys.stdout.reconfigure(encoding='utf-8')  # 确保控制台输出的编码正确

# 加载预训练的 LLaMA 3.2-1B 模型和分词器
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 如果 pad_token 没有设置，则使用 eos_token

# 定义 LoRA 配置用于微调
lora_config = LoraConfig(
    r=8,               # 低秩矩阵的维度
    lora_alpha=32,     # LoRA 的 alpha 超参数
    target_modules=["q_proj", "v_proj"],  # 作用于自注意力层的模块
    lora_dropout=0.05, # 防止过拟合的 dropout 概率
    bias="none"        # 不使用 bias
)

# 将 LoRA 应用于 LLaMA 模型
lora_model = get_peft_model(model, lora_config)

# 加载问答数据集
def load_qa_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        qa_data = f.read().split("--------------------------------------------------------------------------------")
    qa_pairs = []
    for qa in qa_data:
        lines = qa.strip().split("\n")
        if len(lines) >= 3:  # 确保有足够的行提取问题和答案
            question = lines[1].replace("Question:", "").strip()
            answer = lines[2].replace("Answer:", "").strip()
            qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

# 从 web_questions_2000_sample.txt 文件加载数据
qa_pairs = load_qa_data("file\web_questions_2000_sample.txt")
qa_dataset = Dataset.from_dict({"question": [pair["question"] for pair in qa_pairs], "answer": [pair["answer"] for pair in qa_pairs]})

# 定义数据预处理函数，使用明确的指令格式，并使用动态填充
def preprocess_function(examples):
    # 使用动态填充和指令格式："Answer the following question: "
    inputs = tokenizer(["Question: " + question + " Answer:" for question in examples["question"]],
                       padding="max_length", truncation=True, max_length = 20)
    
    labels = tokenizer(examples["answer"], padding="max_length", truncation=True, max_length = 20)["input_ids"]
    
    # 将填充值替换为 -100，以忽略损失计算中的填充部分
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    
    inputs["labels"] = labels
    
    # 打印部分处理后的样本数据
    for i in range(2):  # 打印前两个样本
        print(f"Example {i+1}:")
        print(f"Question (tokenized): {inputs['input_ids'][i]}")
        print(f"Answer (tokenized): {inputs['labels'][i]}")
        print(f"Decoded Question: {tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)}")
        print(f"Decoded Answer: {tokenizer.decode([label for label in inputs['labels'][i] if label != -100], skip_special_tokens=True)}")
        print("\n")
    
    return inputs

# 对数据集进行分词处理
tokenized_dataset = qa_dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./lora_llama_finetune",  # 保存模型的输出目录
    per_device_train_batch_size=4,       # 每个设备的训练批次大小
    per_device_eval_batch_size=2,        # 每个设备的验证批次大小
    num_train_epochs=4,                  # 训练的 epoch 数
    logging_dir="./logs",                # 日志目录
    learning_rate=1e-5,                  # 微调的学习率
)

# 使用 Trainer 初始化训练器，传入 LoRA 模型和微调数据
trainer = Trainer(
    model=lora_model,                    # LoRA 微调后的模型
    args=training_args,                  # 训练参数
    train_dataset=tokenized_dataset,     # 训练数据集
)

# 开始训练
trainer.train()

# 保存微调后的模型
lora_model.save_pretrained("./lora_llama_finetuned_QA")
