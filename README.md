# TESS: A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks

This repository contains the reference implementation for TESS, a scalable temporally and spatially local learning rule for spiking neural networks (SNNs). This work has been accepted for publication in Proceedings of the International Joint Conference on Neural Networks (IJCNN) 2025.

[[ArXiv Paper]](https://arxiv.org/abs/2502.01837) [[IJCNN Paper]](https://doi.org/10.1109/IJCNN64981.2025.11227652)

## Abstract
The demand for low-power inference and training of deep neural networks (DNNs) on edge devices has intensified the need for algorithms that are both scalable and energy-efficient. While spiking neural networks (SNNs) allow for efficient inference by processing complex spatio-temporal dynamics in an event-driven fashion, training them on resource-constrained devices remains challenging due to the high computational and memory demands of conventional error backpropagation (BP)-based approaches. In this work, we draw inspiration from biological mechanisms such as eligibility traces, spike-timing-dependent plasticity, and neural activity synchronization to introduce TESS, a temporally and spatially local learning rule for training SNNs. Our approach addresses both temporal and spatial credit assignments by relying solely on locally available signals within each neuron, thereby allowing computational and memory overheads to scale linearly with the number of neurons, independently of the number of time steps. Despite relying on local mechanisms, we demonstrate performance comparable to the backpropagation through time (BPTT) algorithm, within $\sim1.4$ accuracy points on challenging computer vision scenarios relevant at the edge, such as the IBM DVS Gesture dataset, CIFAR10-DVS, and temporal versions of CIFAR10, and CIFAR100. Being able to produce comparable performance to BPTT while keeping low time and memory complexity, TESS enables efficient and scalable on-device learning at the edge.

<p align = "center">
<img src = "./images/TESS_diagram.png" width="400">
</p>
<p align = "center">
Figure 1: Overview of TESS.
</p>

## ⚠️ Implementation & Reproducibility Note
This repository was recently updated to clearly distinguish between the TESS (Full) and S-TLLR (Baseline) modes.

- S-TLLR Mode: Use `--training-mode s-tllr`. This is a temporal local learning rule baseline.

- TESS Mode: Use `--training-mode tess`. This enables the full spatiotemporal local learning rule.

Note on TESS Convergence: As noted in the paper, TESS is highly sensitive to hyperparameter configurations (normalization, LR balancing, etc.). The current implementation serves as a reference for the logic; users may need to perform further hyperparameter tuning to achieve high performance across different environments.

## How to Use

1. Install the required dependencies listed in `requirements.txt`. 
2. Use the following command to run an experiment:

    ```shell
    python main.py --param-name param_value
    ```

    A description of each parameter is provided in `main.py`.


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{11227652,
  author={Apolinario, Marco P. E. and Roy, Kaushik and Frenkel, Charlotte},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)}, 
  title={TESS: A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks}, 
  year={2025},
  pages={1-9},
  doi={10.1109/IJCNN64981.2025.11227652}}

```
