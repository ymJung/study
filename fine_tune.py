from datasets import load_dataset
import configparser
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


config = configparser.ConfigParser()
config.read('config.cfg')
HF_TOKEN = config['hf']['TOKEN']

model_name = "Qwen/Qwen2.5-Coder-7B"
model_path = "/home/zero00/Dev/datasets/models/qwen2.5-coder-7b"
train_output_dir = model_path + "/results"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=model_path,
    token=HF_TOKEN,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=model_path,
    token=HF_TOKEN,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # qwen modules
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="/home/zero00/Dev/datasets/output/split_dataset.json")

MAX_LENGTH = 256

def tokenize_function(examples):
    combined_texts = [
        f"Project: {project}\n"
        f"Module: {module}\n"
        f"File Name: {file_name}\n"
        f"Content:\n{content}"
        for project, module, file_name, content in zip(
            examples['project_name'],
            examples['module'],
            examples['file_name'],
            examples['content']
        )
    ]
    return tokenizer(
        combined_texts,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)


training_args = TrainingArguments(
    output_dir=train_output_dir,
    evaluation_strategy="no",
    learning_rate=1e-4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    fp16=True,
    optim="paged_adamw_8bit",
    save_total_limit=4,  # 체크포인트 
    save_strategy="steps", 
    save_steps=500,  
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

class CustomTrainer(Trainer):
    def _move_model_to_device(self, model, device): 
        pass

# Trainer 객체 생성
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator
)

# 모델 학습 시작
trainer.train(resume_from_checkpoint=train_output_dir) # checkpoint resume

# LoRA 어댑터 저장
output_model_path = Path(model_path + "/finetuned_qwen_coder_lora")
output_model_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(output_model_path))

print('end')
