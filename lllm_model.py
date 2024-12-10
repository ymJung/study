import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import configparser


# 파인튜닝 시 사용했던 기본값 재사용
model_name = "Qwen/Qwen2.5-Coder-7B"
model_path = "/home/zero00/Dev/datasets/models/qwen2.5-coder-7b"   # 베이스 모델 로컬 경로
peft_model_path = "/home/zero00/Dev/datasets/models/qwen2.5-coder-7b/finetuned_qwen_coder_lora"  # 파인튜닝 결과 LoRA 어댑터 경로
config = configparser.ConfigParser()
config.read('config.cfg')
HF_TOKEN = config['hf']['TOKEN']


# 파인튜닝 시 사용한 quantization_config 복원
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# 토크나이저 로드 (trust_remote_code=True로 파인튜닝 시와 동일하게)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=model_path,
    token=HF_TOKEN,
    trust_remote_code=True
)

# 베이스 모델 로드 (파인튜닝 시와 동일한 매개변수)
# - trust_remote_code=True
# - device_map="auto"
# - quantization_config=bnb_config
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=model_path,
    token=HF_TOKEN,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)



# 파인튜닝 시점에 prepare_model_for_kbit_training를 적용했었으므로,
from peft import prepare_model_for_kbit_training
base_model = prepare_model_for_kbit_training(base_model)

# PEFT(LoRA) 어댑터를 로드
model = PeftModel.from_pretrained(base_model, peft_model_path)
model.eval()

# GPU 사용 가능시 GPU로 이동
if torch.cuda.is_available():
    model.to("cuda")

# 추론 예시
input_text = "def hello_world():"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")

with torch.no_grad():
    outputs = model.generate(input_ids, max_length=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
