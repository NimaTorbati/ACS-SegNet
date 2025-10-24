# ACS-SegNet: An Attention-Based CNN-SegFormer Segmentation Network for Tissue Segmentation in Histopathology
This repository contains the implementation details of "ACS-SEGNET: AN ATTENTION-BASED CNN-SEGFORMER SEGMENTATION NETWORK FOR TISSUE SEGMENTATION IN HISTOPATHOLOGY".

**Acknowledgment**: This project has been conducted through a joint WWTF-funded project (Grant ID: 10.47379/LS23006) between the Medical University of Vienna and Danube Private University. 


## Model
Two Parallel Encoders in a U-Net–Shaped Network:

<img width="4240" height="1268" alt="image" src="https://github.com/user-attachments/assets/906d5d9d-7013-437c-b426-06bc9b2aa87c" />

## Results
Table 1. Segmentation results on the GCPS dataset
| Method          | μIoU (%)         | μDice (%)        |
| --------------- | ---------------- | ---------------- |
| DGAUNet [15]    | 75.95 ± 0.20     | 86.33 ± 0.13     |
| SegFormer [13]  | 70.90 ± 0.38     | 82.97 ± 0.26     |
| ResNetUNet [17] | 75.65 ± 0.08     | 86.13 ± 0.05     |
| TransUNet [6]   | 74.84 ± 0.10     | 85.61 ± 0.09     |
| CS-SegNet       | 76.68 ± 0.15     | 86.80 ± 0.06     |
| **ACS-SegNet**  | **76.79 ± 0.14** | **86.87 ± 0.09** |


Table 2. Segmentation results on the PUMA dataset
| Method          | μIoU (%)         | μDice (%)        |
| --------------- | ---------------- | ---------------- |
| DGAUNet [15]    | 44.35 ± 1.76     | 53.69 ± 1.91     |
| SegFormer [13]  | 46.78 ± 2.58     | 58.25 ± 2.01     |
| ResNetUNet [17] | 58.42 ± 1.98     | 71.58 ± 1.21     |
| TransUNet [6]   | 62.43 ± 2.47     | 74.63 ± 1.91     |
| CS-SegNet       | 63.67 ± 1.23     | 75.55 ± 1.71     |
| **ACS-SegNet**  | **64.93 ± 2.28** | **76.60 ± 1.36** |


## Example Predictions
<p align="center">
  <img width="500" height="1000" alt="image" src="https://github.com/user-attachments/assets/5e3beb8d-590a-4e3e-9fac-1b1297caddd5" />
</p>

## Citation
A preprint version of our paper is publicy available on arXiv: (https://arxiv.org/abs/2510.20754)



BibTex entry:
''' bash
@misc{torbati2025acssegnetattentionbasedcnnsegformersegmentation,
      title={ACS-SegNet: An Attention-Based CNN-SegFormer Segmentation Network for Tissue Segmentation in Histopathology}, 
      author={Nima Torbati and Anastasia Meshcheryakova and Ramona Woitek and Diana Mechtcheriakova and Amirreza Mahbod},
      year={2025},
      eprint={2510.20754},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.20754}, 
}
