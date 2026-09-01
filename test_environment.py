import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "/home/admin01/.cache/modelscope/hub/models/qwen/Qwen1.5-1.8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)

print("LoRA加载成功 ")

inputs = tokenizer("榴莲叶子发黄怎么办？", return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))
