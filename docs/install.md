# Philips pytorch image model zoo
ptimz is a collection of pytorch image deep learning model architectures, losses, metrics, which will help you to setup your own deep learning pipelines with more simple steps.

## Before you install
When using deep learning, you should have:  
- installed nvidia [CUDA](https://developer.nvidia.com/cuda-downloads)/[cudnn](https://developer.nvidia.com/rdp/cudnn-download)/[nccl](https://developer.nvidia.com/nccl/nccl-download).  **It is highly recommended that using GPUs to do deep learning**
- installed python3 (version 3.6/3.7/3.8/3.9)
- setup a virtualenv  


### Install python3
---
**For ubuntu users**
```bash
sudo apt update
sudo apt install python3-dev python3-pip python3-venv

cd "a directory to install your virtualenv"
python3 -m venv --system-site-packages ./ptimzvenv
```

&emsp;

Activate the virtual environment
```
source ./ptimzvenv/bin/activate
```

&emsp;

**For Windows10 users**  
1. Open Microsoft Store and install "Python 3.9"
![ms py39](https://res.cloudinary.com/practicaldev/image/fetch/s--qSFA9T2R--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/i/pevwmsbrna7xs907br3d.png)

2. Open Windows PowerShell as Administrator
3. Setup virtualenv
```
cd "a directory to install your virtualenv"
python3.9 -m venv --system-site-packages ptimzvenv
```
4. Activate the virtual environment
```
.\ptimzvenv\Scripts\Activate.ps1
```



---
## Install ptimz with pip
Use command line, Linux/MacOS terminal or Windows Powershell. And activate your virtualenv

&emsp;

### Install python gpu dependencies.
---
Before installing ptimz cpu version, you can install pytorch and mmcv to GPU version.  
e.g. CUDA-11.6 cudnn-8.4.0 nccl-2.12.10
```bash
CUDA_VERSION=11.6
TORCH_VERSION=1.11.0

# install gpu version pytorch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# Apr 2022 torch version=1.11.0

# install gpu version mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.11.0/index.html
```

&emsp;

### Linux install model zoo package
---
``` bash
pip install https://github.com/songphilips/ptimz/releases/download/v0.0.1-hrnet/ptimz-0.0.1-py3-none-any.whl
```

&emsp;

### Windows install model zoo package
---
1. Download [Model zoo wheel](https://github.com/songphilips/ptimz/releases/download/v0.0.1-hrnet/ptimz-0.0.1-py3-none-any.whl)
2. Copy your local wheel path, e.g. D:\WorkSpace\ptimz-0.0.1-py3-none-any.whl
```
pip install D:\WorkSpace\ptimz-0.0.1-py3-none-any.whl
```