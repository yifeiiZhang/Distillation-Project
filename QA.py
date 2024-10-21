from datasets import load_dataset

# 启用流式加载，避免一次性加载整个数据集
dataset = load_dataset("google-research-datasets/natural_questions", split="train", streaming=True)

# 通过enumerate限制流式数据到1000条
sampled_data = []
for i, item in enumerate(dataset):
    if i >= 1000:  # 只获取前1000条
        break
    sampled_data.append(item)

# 创建并格式化为QA格式
prompts = []
for i, item in enumerate(sampled_data):
    question = item['question']['text']
    
    # 优先选择长答案
    if item['annotations'][0]['long_answer']['start_token'] != -1:
        start_token = item['annotations'][0]['long_answer']['start_token']
        end_token = item['annotations'][0]['long_answer']['end_token']
        answer = " ".join([token['token'] for token in item['document']['tokens'][start_token:end_token]])
    # 如果没有长答案，使用短答案
    elif item['annotations'][0]['short_answers']:
        answer = ", ".join([ans['text'] for ans in item['annotations'][0]['short_answers']])
    # 如果没有长答案或短答案，则检查是否是是/否问题
    elif item['annotations'][0]['yes_no_answer'] != -1:
        answer = "yes" if item['annotations'][0]['yes_no_answer'] == 1 else "no"
    else:
        answer = "无答案"
    
    # 格式化为你需要的QA格式
    prompt = f"---------------------------------------------------\nPrompt {i+1}:\nQuestion: {question}\nAnswer: {answer}\n---------------------------------------------------\n"
    prompts.append(prompt)

# 将prompts写入到txt文件中
output_path = "natural_questions_1000_sample.txt"
with open(output_path, "w") as f:
    f.writelines(prompts)

output_path
