# [*arXiv*] Implicit Reconstruct Spatiotemporal Super-Resolution Microscopy in Arbitrary Dimension

**Motivations**: High-resolution 4D fluorescence microscopy imaging, essential for deciphering dynamic biological processes, is typically challenged by insufficient spatiotemporal resolutions. Phototoxicity, photobleaching, and the anisotropic resolution inherent in current live cell image acquisition systems. To address these challenges, we propose an implicit neural representation-based arbitrary scale super-resolution framework, termed SpatimeINR, which leverages spatiotemporal latent representation in conjunction with a multilayer perceptron for 4D rendering, while incorporating cycle-consistency loss to ensure fidelity with the original data. Extensive experiments on lung cancer cell and C.elegans cell membrane fluorescence datasets demonstrate that our approach significantly outperforms state-of-the-art methods in both temporal and spatial (4D) super-resolution tasks. Ablation studies further confirmed the critical contributions of the spatiotemporal latent representation, 4D rendering, and cycle-consistency loss. Code and data will be released after the review process.


## Overview
<img width="800" alt="Ai" src="https://github.com/user-attachments/assets/40308712-faad-4b06-b41d-3e6acecafa88">  

*** An overview of our proposed SpatimeINR method***
## Get Started
### Dependencies and Installation
- Python 3.11.0
- Pytorch 2.0.1

1. Create Conda Environment
```
conda create --name SpatimeINR python=3.11.0
conda activate SpatimeINR
conda install anaconda::git
```
2. Clone Repo
```
git clone https://github.com/cunminzhao/SpatimeINR.git
```
3. Install Dependencies
```
cd SpatimeINR
pip install -r ./requirement.txt
```

### Dataset
You can refer to the following links to download the datasets of raw cell membrane fluorescence 
[Raw_memb.zip](https://drive.google.com/file/d/1HIFOrZ51F_eN-dybjgZYX4RZrqsnjd7x/view?usp=sharing).

### Training  
**Example**: to train yourdata with *SpatimeINR*, you need to keep these data in
* **Structure of data folder**: 
    ```buildoutcfg
    data/
      |yourdata.nii.gz #data.shape=(x,y,z,t)
    Config/
      |config.yaml
    ```

Then run
```
python train.py --config ./Config/config.yaml --save_path "./save" --file "../data/yourdata.nii.gz"
```

you will get your result in 
* **Structure of save folder**: 
    ```buildoutcfg
    save/
      |current.pth
      |latent_code.pt
    ```


### Inference  
Then run，where ```scale``` is the spatial resolution that you can set arbitrarily, and ```scaleT``` is the temporal resolution that you can set arbitrarily. ```yourdata``` is also necessary for the sampling shape.
```
python inference.py --config ./Config/config.yaml --save_path "./save" --file "./data/yourdata.nii.gz" --scale 1 --scaleT 2
```

you will get your result in 
 ```buildoutcfg
    save/
      |current.pth
      |latent_code.pt
      |output.nii.gz #data.shape=(x*scale,y*scale,z*scale,t*scaleT)
 ``` 

## Compute Report
This method was tested on both the NVIDIA RTX 4090 and A100, and will be released soon.

## Acknowledgement
- We appreciate several previous works for their algorithms and datasets related/helpful to this project, including [*LIIF*](https://github.com/yinboc/liif), [*RCAN*](https://github.com/AiviaCommunity/3D-RCAN), [*CuNeRF*](https://github.com/NarcissusEx/CuNeRF), and [*SVIN*](https://github.com/yyguo0536/SVIN).

