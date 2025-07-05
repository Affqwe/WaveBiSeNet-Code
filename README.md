# WaveBiSeNet

This repository contains the code implementation of WaveBiSeNet

## 📦 Environment
GPU: NVIDIA GeForce RTX 4090 (24 GB)
Python: 3.9.19  
PyTorch: 1.8.1  
CUDA: 12.4
torch==1.8.1
torchvision==0.9.1
numpy>=1.19.5
opencv-python>=4.5.1
scipy>=1.5.4
tqdm>=4.50.0
matplotlib>=3.3.0

Please organize your dataset in the following structure:
data/
├── images/
│   ├── img1.png
│   ├── ...
└── masks/
    ├── mask1.png
    └── ...
Training
python train_u_others.py --data_dir ./data/


