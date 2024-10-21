from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# 加载基础模型 (LLaMA)
base_model_name = "meta-llama/Llama-3.2-1B"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# # 打印基础模型的某个层的权重
print("Base model weights before LoRA:")
# print(base_model.model.layers[0].self_attn.lora_A['default'].weight)
print(base_model.model.layers[0].self_attn.q_proj.weight)
# print(base_model.model.layers[0].self_attn.q_proj.lora_A['default'].weight)
print("Base model structure:")
print(base_model)

# # 加载 LoRA 配置和权重
peft_model_id = "./lora_llama_finetuned"
peft_config = PeftConfig.from_pretrained(peft_model_id)

# # # 打印 LoRA 微调后的权重 (LoRA 权重是一个低秩矩阵)
# print("\nLoRA weights (before applying to the base model):")
lora_model = PeftModel.from_pretrained(base_model, peft_model_id).to("cuda")
print(lora_model.model.model.layers[0].self_attn.q_proj.lora_A['default'].weight)
print(lora_model.model.model.layers[0].self_attn.q_proj.lora_B['default'].weight)

# # 将 LoRA 权重应用到基础模型
model_with_lora = PeftModel.from_pretrained(base_model, peft_model_id).to("cuda")
# print(lora_model.base_model.model)

# 比较前后的权重，看看 LoRA 是否成功应用到基础模型
# # 打印基础模型的结构，查看各层的命名
print("Base model structure:")
print(base_model)

