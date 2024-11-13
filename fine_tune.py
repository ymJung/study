import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from pathlib import Path
from huggingface_hub import login
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch, Accelerator
import configparser

# Config 파일에서 Hugging Face Token 로드
config = configparser.ConfigParser()
config.read('config.cfg')
HF_TOKEN = config['hf']['TOKEN']

# Hugging Face 로그인
try:
    login(token=HF_TOKEN) 
except ImportError:
    print("huggingface_hub 라이브러리가 설치되어 있는지 확인하세요. pip install huggingface_hub 명령을 통해 설치할 수 있습니다.")

# 모델 이름 및 경로 지정
model_name = "meta-llama/CodeLlama-7b-hf"
model_path = "/home/zero00/Dev/datasets/models"

# GPU 또는 MPS 사용 여부 확인
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU: CUDA")
elif torch.has_mps:
    device = torch.device("mps")
    print("Using GPU: MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# 모델 다운로드
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path, use_auth_token=True)

# 장치 맵 설정 및 모델 로드 (GPU와 CPU로 분산)
device_map = infer_auto_device_map(model, max_memory={0: "8GiB", "cpu": "10GiB"})
model = load_checkpoint_and_dispatch(
    model, checkpoint=model_path, device_map=device_map, offload_folder="/home/zero00/Dev/datasets/offload"
)

# JSON 파일 로드
dataset = load_dataset("json", data_files="/home/zero00/Dev/datasets/output/multi_project_dataset.json")

MAX_LENGTH = 1000

# 텍스트 길이를 제한하는 함수
def truncate_text(text, max_length=MAX_LENGTH):
    return text[:max_length]

# 유효한 길이를 확인하는 함수
def is_valid_length(example):
    combined_text = (
        f"Project: {example['project_name']}\n"
        f"Module: {example['module']}\n"
        f"File Name: {example['file_name']}\n"
        f"Content:\n{example['content']}"
    )
    return len(combined_text) <= MAX_LENGTH

# 데이터 토크나이징 함수 (배치 처리)
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
    # 패딩을 max_length로 고정하고 텐서 형식으로 반환
    return tokenizer(
        combined_texts,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 필터링
filtered_dataset = dataset.filter(is_valid_length)

# 데이터셋 토크나이징 (불필요한 컬럼 제거)
tokenized_datasets = filtered_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['file_name', 'file_path', 'module', 'project_name', 'content']
)

# 동적 패딩을 위한 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 데이터셋 분할
split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# 학습 설정 구성 (DeepSpeed 사용)
training_args = TrainingArguments(
    output_dir=model_path + "/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    per_device_train_batch_size=1,
    push_to_hub=False,
    no_cuda=(device != torch.device("cuda")),
    fp16=True,  # Mixed precision 훈련 사용
    deepspeed="ds_config.json"  # DeepSpeed 설정 파일 경로
)

# DeepSpeed 설정 파일 생성
ds_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "train_batch_size": 1
}
import json
with open("ds_config.json", "w") as f:
    json.dump(ds_config, f)

# Accelerator 사용하여 모델 및 데이터 준비
accelerator = Accelerator()
model, train_dataset, eval_dataset = accelerator.prepare(model, train_dataset, eval_dataset)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 모델 학습 시작
trainer.train()

# 모델 저장
output_model_path = Path(model_path + "/finetuned_code_llama")
output_model_path.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(output_model_path))

print('end')
