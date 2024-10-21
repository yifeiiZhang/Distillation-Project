from datasets import load_dataset
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7及以上版本有效

# 加载SQuAD v2 数据集
dataset = load_dataset("rajpurkar/squad_v2")

# 定义生成prompt的函数
def create_prompt(example):
    question = example['question']
    # 如果有答案，使用答案；如果没有，将答案设为"N/A"
    answer = example['answers']['text'][0] if len(example['answers']['text']) > 0 else 'No answer'
    # 格式化成Question-Answer的形式
    return f"Question: {question}\nAnswer: {answer}"

# 随机打乱训练数据，并选择前2000个样本
train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))

# 生成2000个训练数据集中的prompts
train_dataset_with_prompts = train_dataset.map(lambda x: {'prompt': create_prompt(x)})

# 打印生成的prompt集的一部分，查看生成结果
for i in range(5):
    print(f"Prompt {i+1}:")
    print(train_dataset_with_prompts[i]['prompt'])
    print('-' * 80)

# 如果需要保存到文件，使用以下代码：
with open("prompts_without_context_output.txt", "w", encoding="utf-8") as f:
    for i in range(1000):
        f.write(f"Prompt {i+1}:\n")
        f.write(train_dataset_with_prompts[i]['prompt'] + "\n")
        f.write('-' * 80 + "\n")