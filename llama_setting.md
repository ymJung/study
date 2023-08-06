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