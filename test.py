from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig  # 引入 peft 库来加载 LoRA 权重
import time

# 加载基础模型并调试
base_model_name = "meta-llama/Llama-3.2-1B"
try:
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print(f"Base model {base_model_name} and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading base model: {e}")

# 确保 tokenizer 有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载 LoRA 配置和微调后的权重并调试
peft_model_id = "./lora_llama_finetuned_QA"  # 微调后的路径
try:
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(base_model, peft_model_id).to("cuda")
    print(f"LoRA model {peft_model_id} loaded successfully.")
except Exception as e:
    print(f"Error loading LoRA model: {e}")

# 加载QA数据集（已经包含专家提示）并调试
def load_qa_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            qa_data = f.read().split("--------------------------------------------------------------------------------")
        
        qa_pairs = []
        for qa in qa_data:
            lines = qa.strip().split("\n")
            if len(lines) >= 3:  # 确保有足够的行提取问题
                question = lines[2].replace("Question:", "").strip()  # 读取问题
                qa_pairs.append(f"You are an expert of basic knowledge in the world, please answer the following Question using a few words. Question: {question}, Answer:")  # 添加提示和问题
        print(f"Loaded {len(qa_pairs)} QA pairs from {file_path}.")
        return qa_pairs
    except Exception as e:
        print(f"Error loading QA data: {e}")
        return []

# 从指定文件加载问题
qa_file_path = "file/modified_predictions_with_prompt.txt"
prompts = load_qa_data(qa_file_path)

# 只预测第一个问题，并打印所有相关信息
def generate_first_prediction(prompt, model, tokenizer):
    try:
        print(f"Prompt:\n{prompt}")
        # 对问题进行编码
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
        print(f"Tokenized input: {inputs}")
        
        # 生成预测
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
        print(f"Raw model output (token ids): {outputs}")
        
        # 解码生成的预测
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Decoded prediction:\n{prediction}")
        
    except Exception as e:
        print(f"Error generating prediction: {e}")

# 进行第一个预测
if prompts:
    first_prompt = prompts[0]  # 只取第一个问题
    generate_first_prediction(first_prompt, base_model, tokenizer)
else:
    print("No prompts available for prediction.")
