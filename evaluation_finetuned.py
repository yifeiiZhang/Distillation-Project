from sklearn.metrics import f1_score
import re

# 清理和预处理文本的函数
def clean_text(text):
    # 将文本转换为小写并去除首尾空格
    text = text.lower().strip()
    # 移除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 计算F1分数的函数
def compute_f1(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# 计算Exact Match的函数
def compute_exact_match(prediction, ground_truth):
    return int(prediction == ground_truth)

# 从文件中加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().split('--------------------------------------------------------------------------------')
    return [line.strip() for line in data if line.strip()]

# 提取答案
def extract_answers(data):
    answers = []
    for entry in data:
        if "Answer:" in entry:
            answer = entry.split("Answer:")[-1].strip()  # 提取答案
            answers.append(answer)
    return answers

# 加载原始问答和预测文件
prompts_file = 'prompts_without_context_output.txt'  # 参考答案文件
predictions_file = 'finetuned_model_predictions.txt'  # 预测文件

prompts = load_data(prompts_file)
predictions = load_data(predictions_file)

# 提取参考答案和预测答案
reference_answers = extract_answers(prompts)
predicted_answers = extract_answers(predictions)

# 确保参考答案和预测答案数量一致
assert len(reference_answers) == len(predicted_answers), "参考答案与预测答案的数量不一致！"

# 初始化分数
exact_matches = 0
total_f1_score = 0

# 逐个计算F1分数和Exact Match
for ref, pred in zip(reference_answers, predicted_answers):
    clean_ref = clean_text(ref)
    clean_pred = clean_text(pred)

    exact_matches += compute_exact_match(clean_pred, clean_ref)
    total_f1_score += compute_f1(clean_pred, clean_ref)

# 计算总的分数
exact_match_score = (exact_matches / len(reference_answers)) * 100
average_f1_score = (total_f1_score / len(reference_answers)) * 100

print(f"Exact Match: {exact_match_score:.2f}%")
print(f"F1 Score: {average_f1_score:.2f}%")
