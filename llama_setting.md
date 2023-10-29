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



``` sh ubuntu cuda install >> 12.2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.1-535.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.1-535.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```




(llama) ymjung@DESKTOP-A9N1I0N:~/github/GroundingDINO$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jul_11_02:20:44_PDT_2023
Cuda compilation tools, release 12.2, V12.2.128
Build cuda_12.2.r12.2/compiler.33053471_0

conda install:: $ 
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu120


>>> import torch
>>> torch.__version__
'2.0.1+cu117'

cuda 는 12.2 / torch 는 117
버전을 맞춰줘야 한다. https://download.pytorch.org/whl/cu117/torch_stable.html

https://github.com/IDEA-Research/GroundingDINO

cuda 11.3 >>> conda install pytorch==1.9.0
 pip install numpy==1.20.0



# conda install::
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

# 11.7 cuda install::
https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
# torch install::
cuda 11.7 >>  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia


> nvcc - not found
# vi ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH




