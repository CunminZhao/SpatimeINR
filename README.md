# [*arXiv*] Implicit Reconstruct Spatiotemporal Super-Resolution Microscopy in Arbitrary Dimension

**Motivations**: High-resolution 4D fluorescence microscopy imaging, essential for deciphering dynamic biological processes, is typically challenged by insufficient spatiotemporal resolutions. Phototoxicity, photobleaching, and the anisotropic resolution inherent in current live cell image acquisition systems. To address these challenges, we propose an implicit neural representation-based arbitrary scale super-resolution framework, termed SpatimeINR, which leverages spatiotemporal latent representation in conjunction with a multilayer perceptron for 4D rendering, while incorporating cycle-consistency loss to ensure fidelity with the original data. Extensive experiments on lung cancer cell and C.elegans cell membrane fluorescence datasets demonstrate that our approach significantly outperforms state-of-the-art methods in both temporal and spatial (4D) super-resolution tasks. Ablation studies further confirmed the critical contributions of the spatiotemporal latent representation, 4D rendering, and cycle-consistency loss. Code and data will be released after the review process.


## Overview
<img width="800" alt="Ai" src="https://github.com/user-attachments/assets/40308712-faad-4b06-b41d-3e6acecafa88">  

** An overview of our proposed SpatimeINR method***
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
**Example**: to run *EmbSAM* with Emb1_Raw, you need to keep these data in
* **Structure of data folder**: 
    ```buildoutcfg
    data/
      |--Emb1_Raw/*.tif
    ```
* **Structure of confs folder**:  
  The Emb1_CellTracing.csv is the tracing result of cell nucleus fluorescence, saved in confs.
    ```buildoutcfg
    confs/
      |--Emb1_CellTracing.csv
      |--running_Emb1.txt
      ...
      |--other running confs
    ```
Then run
```
python .\EmbSAM.py --config_file ./confs/running_Emb1.txt --nii_path "./nii" --opdir "./output/"
```

you will get your result in 
```
./output_folder/result
```

## Provided Data
* All the 5 embryo samples processed in this paper are digitized into the format customized to our visualization software *ITK-SNAP-CVE* from [https://doi.org/10.1101/2023.11.20.567849](https://doi.org/10.1101/2023.11.20.567849), and can be downloaded [online](https://doi.org/10.6084/m9.figshare.24768921.v2).  

* The effective visualization is shown below:

    *  <img width="900" alt="GUIDATA_SHOW" src="https://github.com/cuminzhao/EmbSAM/assets/80189429/ef30e2dd-e29d-4e0d-bf2f-cdb01b254ed0">  

## Running Time
Following the instructional video on SharePoint[[here](https://innocimda-my.sharepoint.com/:f:/p/zelin/EiawdEXlPAxJk98XbtPqfgUBt_SMlnbgE9VmnLChHSiXJQ?e=TNAfFu)] and YouTube[[here](https://www.youtube.com/watch?v=sir2-IMKDwU)] (provided in Google Colab[[here](https://drive.google.com/file/d/1CA3g2WEhPmwvSzE_QL8wBkd3nD_XWMCE/view?usp=drive_link)], which is accessible through figshare[[here](https://doi.org/10.6084/m9.figshare.29064530)]), two individuals uninvolved in EmbSAM development were able to execute the complete procedure (incl., downloading the compressed raw image, installing the required Python packages via pip, running the full workflow, and completing segmentation of a small, representative test dataset) in <1 hour. Once the preparation was aaccomplished, the computational time for processing actual image stacks — dependent on time points, processing unit, and so forth — was evaluated (see Supplemental Data 3 of the paper) as <6 minutes per time point.

## Acknowledgement
- We appreciate several previous works for their algorithms and datasets related/helpful to this project, including [*CShaper*](https://doi.org/10.1038/s41467-020-19863-x), [*CMap*](https://doi.org/10.1101/2023.11.20.567849), [*LLFlow*](
https://doi.org/10.48550/arXiv.2109.05923), [*3DMMS*](https://doi.org/10.1186/s12859-019-2720-x), [*MedLSAM*](
https://doi.org/10.48550/arXiv.2306.14752), and [*Segment Anything*](
https://doi.org/10.48550/arXiv.2304.02643).

