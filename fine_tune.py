from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import configparser
from pathlib import Path
import torch

# Config 파일에서 Hugging Face Token 로드
config = configparser.ConfigParser()
config.read('config.cfg')
HF_TOKEN = config['hf']['TOKEN']

# 모델 이름 및 경로 지정
model_name = "meta-llama/CodeLlama-7b-hf"
model_path = "/home/zero00/Dev/datasets/models"

# 모델 및 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=model_path,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16  # Mixed Precision 사용
)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# JSON 파일 로드 및 토크나이징
dataset = load_dataset("json", data_files="/home/zero00/Dev/datasets/output/multi_project_dataset.json")
MAX_LENGTH = 1000

def truncate_text(text, max_length=MAX_LENGTH):
    return text[:max_length]

def is_valid_length(example):
    combined_text = (
        f"Project: {example['project_name']}\n"
        f"Module: {example['module']}\n"
        f"File Name: {example['file_name']}\n"
        f"Content:\n{example['content']}"
    )
    return len(combined_text) <= MAX_LENGTH

def tokenize_function(examples):
    combined_texts = [
        f"Project: {truncate_text(project)}\n"
        f"Module: {truncate_text(module)}\n"
        f"File Name: {truncate_text(file_name)}\n"
        f"Content:\n{truncate_text(content)}"
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

filtered_dataset = dataset.filter(is_valid_length)
tokenized_datasets = filtered_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['file_name', 'file_path', 'module', 'project_name', 'content']
)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir=model_path + "/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    per_device_train_batch_size=1,  # Adjust as per GPU memory
    gradient_accumulation_steps=4,
    fp16=True,
    deepspeed="ds_config.json"  # DeepSpeed 설정 파일 경로
)

# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"].train_test_split(test_size=0.1)["test"],
    data_collator=data_collator
)

# 모델 학습 시작
trainer.train()

# 모델 저장
output_model_path = Path(model_path + "/finetuned_code_llama")
output_model_path.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(output_model_path))

print('end')
