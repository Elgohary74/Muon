# Muon: MoE Training with Newton-Schulz Optimizer

This repository implements a **Mixture of Experts (MoE)** transformer model trained using the **Muon optimizer**. The Muon optimizer utilizes Newton-Schulz iterations for preconditioning efficiently on the GPU, designed to optimize large-scale neural network training.

##  Key Features

*   **Muon Optimizer**: Incorporates the custom `Muon` optimizer (Zero-Power via Newton-Schulz) for weight updates, potentially offering faster convergence or better stability for certain architectures.
*   **Mixture of Experts (MoE)**: Train sparse MoE models with configurable experts, top-k routing, and load balancing.
*   **Optimized Training**: Built on PyTorch with support for CUDA, mixed precision (AMP), and optimized data loading.
*   **Dataset Integration**: Seamlessly integrates with HuggingFace Datasets (using `cosmopedia-v2` by default).

## how to run 
run our colab notebook https://github.com/Elgohary74/Muon/blob/main/Muon_experiment.ipynb ( ensure you are using gpu )

## resources 
Muon : 
Blog/Paper: Muon: https://kellerjordan.github.io/posts/muon/

Video Explanation: https://www.youtube.com/watch?si=xY1zOTrSmRSgDrEh&v=bO5nvE289ec&feature=youtu.be

Applied Research: Muon is Scalable for LLM Training( paper we applied ) : https://arxiv.org/pdf/2502.16982 

you must underatnd mixture of expert and it is importnat for llms training and infernce : 

video : https://www.youtube.com/watch?v=Fg8urTOImpY&t=994s 

papers :
Sparse Mixture of Experts paper: https://arxiv.org/abs/1701.06538 

Mixtral of Experts: https://arxiv.org/abs/2401.04088 

DeepSeek V2: https://arxiv.org/abs/2405.04434 

DeepSeek V3: https://arxiv.org/abs/2412.19437 

Switch Transformers / Expert Capacity:  https://arxiv.org/abs/2101.03961 

Implementation resources :

DeepSeek-V3 Codebase: https://github.com/deepseek-ai/DeepSeek-V3/tree/main

Moonlight: https://github.com/MoonshotAI/Moonlight

Model: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2

Dataset: HuggingFaceTB/cosmopedia-v2 


## future work : 
try muon on other model ( i will discuss with you later ) 

try to solve muon fine-tuning problem 
