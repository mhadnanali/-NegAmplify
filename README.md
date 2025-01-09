# NegAmplify
## From Overfitting to Robustness: Quantity, Quality, and Variety Oriented Negative Sample Selection in Graph Contrastive Learning
### Paper URL [https://www.sciencedirect.com/science/article/pii/S1568494624014467](url)


## Requirements

- **Python**: 3.10.4
- **PyTorch**: 1.11.0+cu113
- **PyGCL**: 0.1.2
- **PyTorch-geometric**: 2.1.0.post1


## Execute Code 
```bash
python NegAmplify.py
```

## Hyper Parameter Settings

| Dataset   | $E_d^1$ | $F_m^1$ | $E_d^2$ | $F_m^2$ | $\tau$ | Training epochs | Learning rate    | Weight decay | Encoder layers | Torch Seeds | Random Seeds |
|-----------|---------|---------|---------|---------|--------|-----------------|------------------|--------------|----------------|-------------|--------------|
| Cora      | 0.45    | 0.35    | 0.15    | 0.5     | 0.4    | 1200            | $5 \times 10^{-3}$ | $1^{-5}$    | 128, 128       |  6521       | 9134         |
| CiteSeer  | 0.95    | 0.85    | 0.3     | 0.25    | 0.2    | 1200            | $5 \times 10^{-4}$ | $1^{-5}$    | 128, 128       |  4122       | 9642         |
| PubMed    | 0.5     | 0.45    | 0.4     | 0.4     | 0.1    | 2000            | $5 \times 10^{-4}$ | $1^{-5}$    | 128, 128       |  6521       | 9134         |
| DBLP      | 0.5     | 0.25    | 0.3     | 0.45    | 0.5    | 1200            | $5 \times 10^{-4}$ | $1^{-5}$    | 128, 128       |  6521       | 9134         |
| WikiCS    | 0.35    | 0.25    | 0.75    | 0.4     | 0.85   | 1200            | $5 \times 10^{-4}$ | $1^{-5}$    | 256, 256       |  3227       | 21895        |
| Am.Comp   | 0.5     | 0.4     | 0.15    | 0.25    | 0.15   | 1200            | $5 \times 10^{-4}$ | $1^{-5}$    | 256, 256       |  5614       | 88224        |
| Am.Photo  | 0.1     | 0.15    | 0.45    | 0.2     | 0.5    | 1200            | $1 \times 10^{-5}$ | $1^{-5}$    | 256, 256       |  67730      | 43158        |
| Co.CS     | 0.2     | 0.5     | 0.5     | 0.4     | 0.7    | 1200            | $1 \times 10^{-5}$ | $1^{-5}$    | 256, 256       |  72010      | 51958        |
| Actor     | 0.5     | 0.35    | 0.3     | 0.5     | 0.2    | 1200            | $5 \times 10^{-4}$ | $1^{-5}$    | 128, 128       |  6672       | 4508         |




## Cite paper as: 
```bibtex
@article{NegAmplify2025,
title = {From overfitting to robustness: Quantity, quality, and variety oriented negative sample selection in graph contrastive learning},
journal = {Applied Soft Computing},
pages = {112672},
year = {2025},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2024.112672},
url = {https://www.sciencedirect.com/science/article/pii/S1568494624014467},
author = {Adnan Ali and Jinlong Li and Huanhuan Chen and Ali Kashif Bashir},
keywords = {Self-supervised learning, Graph representation learning, Graph contrastive learning, Deep learning, Negative sampling}
}
```
