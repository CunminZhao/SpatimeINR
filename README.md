# [*arXiv*] *EmbSAM*: Cell boundary localization and Segment Anything Model for 3D fast-growing embryos

**Motivations**: Cellular shape dynamics are critical for understanding cell fate determination and organogenesis during development. However, fluorescence live-cell images of cell membranes frequently suffer from a low signal-to-noise ratio, especially during long-duration imaging with high spatiotemporal resolutions. This is caused by phototoxicity and photobleaching, which limit laser power and hinder effective time-lapse cell shape reconstruction, particularly in rapidly developing embryos.  

**Results**: We devised a new computational framework, EmbSAM, that incorporates a deep-learning-based cell boundary localization algorithm and the Segment Anything Model. EmbSAM enables accurate and robust three-dimensional (3D) cell membrane segmentation for roundworm Caenorhabditis elegans embryos imaged every 10 seconds. The cell shape data prior to gastrulation quantitatively characterizes a series of cell-division-coupled morphodynamics associated with cell position, cell identity, lineage, and fate, and can be accessed locally and online. The framework also exhibits potential in segmenting and quantifying the fluorescence labeling various cell-membraned-attached molecules in both wild-type and RNAi-treated embryos.


## Overview
<img width="900" alt="Ai" src="https://github.com/user-attachments/assets/eb76f337-db21-4e6c-9b14-f51ac73e8439">     

**Fig. 1. The flowchart of *EmbSAM***
## Get Started
### Dependencies and Installation
- Python 3.11.0
- Pytorch 2.0.1

1. Create Conda Environment
```
conda create --name EmbSAM python=3.11.0
conda activate EmbSAM
conda install anaconda::git
```
2. Clone Repo
```
git clone https://github.com/cunminzhao/EmbSAM.git
```
3. Install Dependencies
```
cd EmbSAM
pip install -r ./requirement.txt
```

### Dataset
You can refer to the following links to download the datasets of raw cell membrane fluorescence 
[Raw_memb.zip](https://drive.google.com/file/d/1HIFOrZ51F_eN-dybjgZYX4RZrqsnjd7x/view?usp=sharing).


### Pre-trained Model
the pre-trained models of this project can be downloaded here
- The model_parameters with Segment Anything Model(vit_b) and the image denosing module trained on X_axis, Y_axis and Z_axis [[model_parameters](https://drive.google.com/file/d/1D9S9Prl60RLF2ftX7dcCc-LTE_tKIYWQ/view?usp=sharing)]. You need to download these parameters and put them into model_parameters.
* **Structure of model_parameters**: 
    ```buildoutcfg
    model_parameters/
      |--Z_axis.pth
      |--Y_axis.pth
      |--X_axis.pth
      |--sam_vit_b_01ec64.pth
    ```

### Test  
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

