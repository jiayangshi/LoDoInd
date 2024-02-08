# LoDoInd

This repository accompanies the paper "[LoDoInd: Introducing A Benchmark Low-dose Industrial CT Dataset and Enhancing Denoising with 2.5D Deep Learning Techniques.](https://www.ndt.net/article/ctc2024/papers/Contribution_187.pdf)"

## Abstract
Computed Tomography (CT) is a widely employed non-destructive testing tool. In industrial applications, minimizing scanning time is crucial for efficient in-line inspection. One approach to achieve faster scanning is through low-dose CT. However, the reduction in radiation dose results in increased noise levels in the reconstructed CT images. Deep learning-based post-processing methods have shown promise in mitigating this noise, but their effectiveness relies on access to datasets with a substantial amount of training data. 

Existing low-dose CT datasets either are not specifically tailored for industrial applications or are based on simulated image formation. In this study, we present a new benchmark low-dose CT dataset, **LoDoInd**, which consists of experimental low-dose CT images explicitly designed for industrial purposes. **LoDoInd** incorporates complex and diverse secondary filling objects within the same testing object, simulating real-world scenarios encountered in industrial settings. The dataset can be accessed at [this Zenodo repository](https://zenodo.org/records/10356955).

Building upon the foundation set by **LoDoInd**, we further investigate the efficacy of various post-processing methods in denoising tasks. Through a detailed comparative analysis of 2D, 2.5D, and 3D training, we demonstrate that 2.5D training strikes an optimal balance between performance and computational efficiency.

## Dataset
**LoDoInd** comprises five distinct low-dose CT reconstructions of a test object, each at different noise levels, along with a reference dataset. The image below showcases a sample slice across various noise levels and the reference:


![noise levels](imgs/noiselevels.png)

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/) should be installed on your system.

## Installation

1. Clone this repository:
   ```bash
   git clone git@github.com:jiayangshi/LoDoInd.git
   cd LoDoInd
   ```

2. Create and activate a Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate denoise-ct
    ```

## How to Use

### Steps
1. **Data Preparation**: Edit [split_train_test.py](split_train_test.py) to specify the noise level and the train/test data ratio. 

2 **Training and Testing**: Modify the training file path in train.py and select the training mode ('2D', '2.5D', '3D').

- For 2D training, set mode to '2D':
```python
    python train.py
```

- For 2.5D training, set mode to '2.5D' (stack size is adjustable, default is 5):
```python
    python train.py
```

- For 3D training, set mode to '3D':
```python
    python convert3h5.py
    python train.py
```
