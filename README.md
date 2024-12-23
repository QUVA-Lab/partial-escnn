# *Partial-escnn* (WIP)
This repository is an uncleaned and preliminary PyTorch implementation of the method proposed in the paper [A Probabilistic Approach to Learning the Degree of Equivariance in Steerable CNNs](https://arxiv.org/abs/2406.03946), accepted at ICML 2024 (see reference below).
This paper presents a novel approach to create Steerable CNNs with layer-wise learnable degree of $(\mathbb{R}^n, +) \rtimes H$ equivariance, where $H\leq O(n)$ is learnable.


The implementations in this repository are based on the [*escnn*](https://github.com/QUVA-Lab/escnn) library, which is a PyTorch library to create steerable CNNs.

# Requirements
Basic requirements
```
Python >= 3.10
torch
torchvision
escnn
```
Requirements for reproduction of all experiments/plots
```
wandb
plotly
matplotlib
seaborn
sklearn
pandas
```
# Reference
Paper accepted at ICML 2024
```
@article{veefkind2024probabilistic,
  title={A Probabilistic Approach to Learning the Degree of Equivariance in Steerable CNNs},
  author={Veefkind, Lars and Cesa, Gabriele},
  journal={arXiv preprint arXiv:2406.03946},
  year={2024}
}
```

