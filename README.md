# ACS-SegNet
This repository contains the implementation details of "ACS-SEGNET: AN ATTENTION-BASED CNN-SEGFORMER SEGMENTATION NETWORK FOR TISSUE SEGMENTATION IN HISTOPATHOLOGY".

Acknowledgment: This project has been conducted through a joint WWTF-funded project (Grant ID: 10.47379/LS23006) between the Medical University of Vienna and Danube Private University. 

## Model Structure
Block diagram of the proposed model. The model consists of two parallel encoders: a SegFormer encoder and a ResNet encoder. Features from the four stages of the SegFormer encoder are first upsampled and concatenated (C. blocks) with the corresponding ResNet encoder features, and then fused through the Convolutional Block Attention Module (CBAM). The fourth block of the ResNet encoder serves as the bottleneck. The decoder is a traditional U-Net decoder with five stages and four skip connections. The SegFormer encoder and ResNet encoder blocks are taken from the SegFormer and ResNet models, respectively. The decoder blocks follow the standard U-Net design. For each path, the feature maps are illustrated with their corresponding dimensions.
<img width="4432" height="2688" alt="image" src="https://github.com/user-attachments/assets/8e821279-f7a9-4745-a7a5-bb9b7a8c5a2b" />

## Results
<img width="3548" height="2448" alt="image" src="https://github.com/user-attachments/assets/173c4685-0019-4783-b7de-86e8cf164310" />


<img width="444" height="231" alt="image" src="https://github.com/user-attachments/assets/f6c91862-519f-41b4-a594-49effd7ce851" />

<img width="460" height="223" alt="image" src="https://github.com/user-attachments/assets/748820a6-165a-475f-994b-22291dc8f45d" />
