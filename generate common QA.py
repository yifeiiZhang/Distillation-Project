import random
from datasets import load_dataset

# 加载WebQuestions数据集
dataset = load_dataset("Stanford/web_questions", split="train")

# 随机抽取2000条数据
sampled_data = random.sample(list(dataset), 2000)

# 创建并格式化为自增ID 问题 回答格式
prompts = []
for i, item in enumerate(sampled_data):
    question_id = i + 1  # 生成自增的ID
    question = item['question']
    
    # WebQuestions中的答案通常是列表，选择第一个答案作为参考
    answer = item['answers'][0] if len(item['answers']) > 0 else "NA"
    
    # 格式化为你需要的prompt格式并增加80个“-”作为分隔线
    prompt = f"ID: {question_id}\nQuestion: {question}\nAnswer: {answer}\n{'-' * 80}\n"
    prompts.append(prompt)

# 将prompts写入到txt文件中，指定utf-8编码
output_path = "web_questions_2000_sample.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(prompts)

print(f"Prompts written to {output_path}")
