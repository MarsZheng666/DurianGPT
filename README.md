# 榴莲GPT 训练项目

完整的榴莲种植大模型微调方案，基于 Qwen3-14B + LoRA/QLoRA。

## 功能特性

### Level 1（强烈建议）
-  Validation dataset 自动分割
-  Early stopping 防止过拟合
-  多种数据格式支持（QA、多轮对话、Reasoning）

### Level 2（推荐）
-  Packing 提升训练效率
-  Flash Attention 加速推理
-  4bit/8bit 量化支持

### Level 3（进阶）
-  多轮对话训练
-  Reasoning 思维链数据支持
-  SFT 标签遮罩

### Level 4（工业级）🚧
-  RLHF / DPO（后续版本）
-  RAG + 微调结合（后续版本）

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 准备数据

#### 方式A：使用API生成QA数据

```bash
export API_KEY="你的API_KEY"
python generate_qa_dataset.py
```

输出：`qa_dataset.jsonl`

#### 方式B：使用现有数据

支持的格式：

**格式1：QA对**
```json
{"question": "...", "answer": "..."}
```

**格式2：多轮对话**
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**格式3：Reasoning（思维链）**
```json
{"question": "...", "reasoning": "...", "answer": "..."}
```

**格式4：Alpaca**
```json
{"instruction": "...", "input": "...", "output": "..."}
```

### 3. 训练模型

#### 基础训练（推荐）

```bash
python train_qwen3_14b_lora_v2.py \
  --model_name_or_path /path/to/Qwen3-14B \
  --train_data_path qa_dataset.jsonl \
  --output_dir ./durian_qwen3_14b_lora_v2 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --use_flash_attention True \
  --early_stopping_patience 3
```

#### 启用Packing（提升效率）

```bash
python train_qwen3_14b_lora_v2.py \
  --model_name_or_path /path/to/Qwen3-14B \
  --train_data_path qa_dataset.jsonl \
  --use_packing True \
  --pack_length 1536
```

#### 自定义验证集

```bash
python train_qwen3_14b_lora_v2.py \
  --model_name_or_path /path/to/Qwen3-14B \
  --train_data_path train.jsonl \
  --val_data_path val.jsonl
```

#### 8bit量化（节省显存）

```bash
python train_qwen3_14b_lora_v2.py \
  --model_name_or_path /path/to/Qwen3-14B \
  --train_data_path qa_dataset.jsonl \
  --use_4bit False \
  --use_8bit True
```

## 参数说明

### 模型参数
- `model_name_or_path`: 模型路径或HF模型ID

### 数据参数
- `train_data_path`: 训练数据路径（JSONL）
- `val_data_path`: 验证数据路径（可选，不指定则自动分割）
- `val_split_ratio`: 验证集比例（默认0.1）
- `max_length`: 最大序列长度（默认1536）

### 训练参数
- `num_train_epochs`: 训练轮数（默认3）
- `per_device_train_batch_size`: 批次大小（默认1）
- `gradient_accumulation_steps`: 梯度累积步数（默认16）
- `learning_rate`: 学习率（默认1e-4）
- `warmup_ratio`: 预热比例（默认0.03）
- `early_stopping_patience`: Early stopping耐心值（默认3）

### LoRA参数
- `lora_r`: LoRA rank（默认16）
- `lora_alpha`: LoRA alpha（默认32）
- `lora_dropout`: LoRA dropout（默认0.05）

### 量化参数
- `use_4bit`: 4bit量化（默认True）
- `use_8bit`: 8bit量化（默认False）

### 高级参数
- `use_flash_attention`: Flash Attention（默认True）
- `use_packing`: Packing（默认False）

## 输出结构

```
durian_qwen3_14b_lora_v2/
├── adapter_config.json
├── adapter_model.bin
├── config.json
├── generation_config.json
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

## 推理

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_id = "./durian_qwen3_14b_lora_v2"
model = AutoPeftModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "你是榴莲种植专家..."},
    {"role": "user", "content": "榴莲叶片黄化怎么办？"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=220, temperature=0.7)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

## 常见问题

### Q: 显存不足怎么办？
A: 尝试以下方案：
1. 减小 `per_device_train_batch_size`（默认1）
2. 增加 `gradient_accumulation_steps`
3. 启用 `use_4bit` 或 `use_8bit`
4. 启用 `use_packing`

### Q: 训练太慢怎么办？
A: 
1. 启用 `use_packing`
2. 启用 `use_flash_attention`
3. 增加 `per_device_train_batch_size`（如果显存允许）
4. 减小 `max_length`

### Q: 模型过拟合怎么办？
A:
1. 增加 `early_stopping_patience`
2. 增加训练数据量
3. 增加 `lora_dropout`
4. 减小 `learning_rate`

### Q: 如何合并LoRA权重？
A:
```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("./durian_qwen3_14b_lora_v2")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./durian_qwen3_14b_merged")
```

## 文件说明

- `train_qwen3_14b_lora.py`: 基础版训练脚本
- `train_qwen3_14b_lora_v2.py`: 增强版（推荐使用）
  - 支持 Validation dataset + Early stopping
  - 支持 Packing + Flash Attention
  - 支持多轮对话和Reasoning数据
- `generate_qa_dataset.py`: API调用生成QA数据
- `requirements.txt`: 依赖包列表

## 下一步

- [ ] 实现RLHF/DPO（Level 4）
- [ ] 集成RAG检索增强
- [ ] 部署推理服务
- [ ] 评估和基准测试
