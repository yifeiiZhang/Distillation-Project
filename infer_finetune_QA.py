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
                qa_pairs.append(f"You are an expert. Please answer the following question in detail.\nQuestion: {question}, Answer:")  # 添加提示和问题
        print(f"Loaded {len(qa_pairs)} QA pairs from {file_path}.")
        return qa_pairs
    except Exception as e:
        print(f"Error loading QA data: {e}")
        return []

# 从指定文件加载问题
qa_file_path = "file/modified_predictions_with_prompt.txt"
prompts = load_qa_data(qa_file_path)

# 用模型生成预测并调试
def generate_predictions(prompts, model, tokenizer):
    predictions = []
    start_time = time.time()  # 记录开始时间
    for prompt in tqdm(prompts, desc="Processing Prompts"):  # 使用tqdm来显示进度
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
            # 设置 pad_token_id 为 eos_token_id
            outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id, num_beams=10,
    temperature=0.7)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
        except Exception as e:
            print(f"Error generating prediction for prompt: {prompt[:50]}... Error: {e}")
    end_time = time.time()  # 记录结束时间
    prediction_time = end_time - start_time  # 计算预测时间
    print(f"Total prediction time: {prediction_time:.2f} seconds")
    return predictions, prediction_time

# 生成预测
if prompts:
    predictions, prediction_time = generate_predictions(prompts, model, tokenizer)

    # 将预测结果保存到文件并调试
    output_path = "file/llama_finetuned_model_predictions2.txt"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred + "\n")
                f.write('-' * 80 + "\n")  # 在每个预测结果之间添加分隔符
        print(f"Predictions saved to {output_path}, time: {prediction_time:.2f} seconds.")
    except Exception as e:
        print(f"Error saving predictions to file: {e}")
else:
    print("No prompts available for prediction.")
