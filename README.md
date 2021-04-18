# Slot-Gated Modeling Using PyTorch

In [Slot-Gated Modeling for Joint Slot Filling and Intent Prediction by Goo et al., 2018](https://www.csie.ntu.edu.tw/~yvchen/doc/NAACL18_SlotGated.pdf), the source code is only given in tensorflow. This repository aims to recreate the model using PyTorch.

## Requirements
* python 3.9
* torch==1.8.1
* for CPU: `pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
* for CUDA 11.1: `pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
* for CUDA 10.2: `pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

## Usage
`python train.py --dataset=atis`