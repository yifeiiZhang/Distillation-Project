from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm  # 引入tqdm库
import time

# 加载未微调的LLaMA模型和分词器
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载QA数据集（仅包含问题）
def load_qa_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        qa_data = f.read().split("--------------------------------------------------------------------------------")
    qa_pairs = []
    for qa in qa_data:
        lines = qa.strip().split("\n")
        if len(lines) >= 3:
            question = lines[1].replace("Question:", "").strip()
            qa_pairs.append(question)
    print(f"Loaded {len(qa_pairs)} QA pairs.")
    return qa_pairs

# 加载刚才生成的文件
prompts = load_qa_data("file\web_questions_2000_sample.txt")

# 用模型生成预测
def generate_predictions(prompts, model, tokenizer):
    predictions = []
    start_time = time.time()
    for prompt in tqdm(prompts, desc="Processing Prompts"):  # 使用tqdm显示进度
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
        # 设置 pad_token_id 为 eos_token_id
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
    end_time = time.time()
    prediction_time = end_time - start_time  # 计算预测时间
    print(f"Total prediction time: {prediction_time:.2f} seconds")
    return predictions, prediction_time

# 生成预测
predictions, prediction_time = generate_predictions(prompts, model, tokenizer)

# 将预测结果保存到文件
output_path = "llama_predictions_from_web_questions.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for pred in predictions:
        f.write(pred + "\n")
        f.write('-' * 80 + "\n")
print(f"Predictions saved to {output_path}, time: {prediction_time:.2f} seconds.")
