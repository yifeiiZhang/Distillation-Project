from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm 
import time

# 加载微调后的模型和分词器
model_path = "./lora_llama_finetuned"
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载QA数据集（仅包含问题和答案）
def load_qa_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        qa_data = f.read().split("--------------------------------------------------------------------------------")
    qa_pairs = []
    for qa in qa_data:
        lines = qa.strip().split("\n")
        if len(lines) >= 2:
            question = lines[1].replace("Question:", "").strip()
            answer = lines[2].replace("Answer:", "").strip()  # 答案不参与预测，只保留问题
            qa_pairs.append(f"Question: {question}, Answer:") 
    print(f"Loaded {len(qa_pairs)} QA pairs.")
    return qa_pairs

prompts = load_qa_data("prompts_without_context_output.txt")

# 用模型生成预测
def generate_predictions(prompts, model, tokenizer):
    predictions = []
    start_time = time.time()  # 记录开始时间
    for prompt in tqdm(prompts, desc="Processing Prompts"):  # 使用tqdm来显示进度
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
        # 设置 pad_token_id 为 eos_token_id
        outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
    end_time = time.time()  # 记录结束时间
    prediction_time = end_time - start_time  # 计算预测时间
    print(f"Total prediction time: {prediction_time:.2f} seconds")
    return predictions, prediction_time

# 生成预测
predictions, prediction_time = generate_predictions(prompts, model, tokenizer)

# 将预测结果保存到文件
with open("finetuned_model_predictions.txt", "w", encoding="utf-8") as f:
    for pred in predictions:
        f.write(pred + "\n")
        f.write('-' * 80 + "\n")  # 在每个预测结果之间添加分隔符

print(f"finetuned_model_predictions.txt, time: {prediction_time:.2f} seconds.")
