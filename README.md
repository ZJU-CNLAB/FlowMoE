# FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Expert Training #  
## Introduction ##
This repository contains the codes of the FlowMoE paper submitted to *NeurIPS 2025*. FlowMoE is a scalable, generic and user-friendly pipeline scheduling framework for accelerating the training of MoE models. FlowMoE outperforms the state-of-the-art scheduling frameworks, including [ScheMoE](https://github.com/Fragile-azalea/ScheMoE), [Tutel](https://github.com/microsoft/tutel) and [FasterMoE](https://github.com/thu-pacman/FasterMoE).  
<div align=center><img src="workflow_nips.png" width="700"/></div> 

## Installation ##
### Prerequisites ###
The following prerequisites shoud be installed for this repository:  
* CUDA >= 11.3  
* PyTorch >= 1.12.1
### How to insatll ###
You can run the following scripts:  
```
# Install zfp
git clone https://github.com/Fragile-azalea/zfp.git
cd zfp
mkdir build
cd build
cmake ..
cmake --build . --config Release
ctest
cd ../..

git clone https://github.com/Fragile-azalea/ScheMoE.git
cd ScheMoE
python setup.py install
```
### Quick start ###
You can download this code to /root/code folder and run the following scripts:  
```
# Single Machine:
cd /root/code/flowmoe/dist_train  
python3 -m torch.distributed.run --nproc_per_node=4 -m train_w_FlowMoE_BO --a2a_ffn_overlap_degree=2 --num_steps=10
```  
Assume that you have 4 GPUs on a single node and everything works well, you will see that there are 4 workers running at a single node training the customized MoE layers using the FlowMoE framework.
```
# Distribute:
# pls refers to flowmoe/dist_train/run_mpi.sh
```
### Test Environment ###
* g++ == 7.5.0  
* cuda == 11.3
* gpu == 3090
