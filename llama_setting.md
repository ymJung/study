 llama - text-gen ui 사용

anaconda 설치
windows 환경
nvdia GPU 사용

허깅페이스 가입
token 발급 https://huggingface.co/settings/tokens
시스템 변수 추가 
HF_USER ::: textgen
HF_PASS ::: {}

git clone https://github.com/oobabooga/text-generation-webui.git

cd .\text-generation-webui\
conda create -n textgen python=3.10.9
conda activate textgen
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r .\requirements.txt
pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl

python server.py >> http://localhost:7860/?__theme=dark

>> MODEL 탭 이동, 허깅페이스의 모델명 입력 meta-llama/Llama-2-13b-hf
- OOM Error -> --auto-devices --chat --gpu-memory 11



-- 파인튜닝하려는데, 윈도우는 너무 버그가 많아 도커 설치중
# powershell >> 
    > wsl --install
    > wsl --set-default-version 2
https://www.docker.com/products/docker-desktop/
    - docker windows hypervisor is not present >> 가상os 를 활성화 해야한다. 
        - BIOS > cpu advanced > SVM Mode > Enable




sudo apt-get update

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"

sudo apt-get install -y docker-ce docker-ce-cli containerd.io

sudo service docker start

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list


 (xx) sudo apt-get install -y nvidia-container-toolkit
 (oo) sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi




torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir /home/ymjung/github/models/meta-llama_Llama-2-7b-chat-hf/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4

``` py
from transformers import AutoTokenizer
import transformers
import torch

model = "/home/ymjung/github/models/meta-llama_Llama-2-7b-chat-hf/"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def send_msg(msg):
    sequences = pipeline(
        msg += '\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1000,
    )
    res = ''
    for seq in sequences:
        res += seq['generated_text']
    return res
```

# 트레이닝 시키기.
pip install trl
git clone https://github.com/lvwerra/trl

python trl/examples/scripts/sft_trainer.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --load_in_4bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2


# ko-alpaca - llama2 
pip install -q autotrain-advanced
autotrain setup --update-torch

!autotrain llm --train \
    --project_name "ko-llama2-finetune" \
    --model "TinyPixel/Llama-2-7B-bf16-sharded" \
    --data_path "royboy0416/ko-alpaca" \
    --text_column "text" \
    --use_peft \
    --use_int4 \
    --learning_rate 2e-4 \
    --train_batch_size 16 \
    --num_train_epochs 2 \
    --trainer sft \
    --model_max_length 2048


https://github.com/huggingface/autotrain-advanced/blob/f1367b590dfc53d240e9684779991da540590386/src/autotrain/cli/run_llm.py#L21



jupyter notebook permissionerror
>> export XDG_RUNTIME_DIR=""



# module ffmpeg has no attribute error
pip install ffmpeg
pip uninstall ffmpeg -y
pip install python-ffmpeg
pip uninstall python-ffmpeg -y
pip install ffmpeg-python
pip uninstall ffmpeg-python -y 로 한 후
pip install ffmpeg-python
