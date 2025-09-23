# MFHS
This is the repository for MFHS : Mutual Consistency Learning-based foundation model integrates Hypergraph Computation for Semi-supervised Medical Image Segmentation  

![framework](pictures/framework.png)
## 👉 Installation
1.Creat a environment  `conda create -n MFHS`  and activate it  `conda activate MFHS`  
2.`git clone https://github.com/Zhai-Mao/MFHS`  
3.Enter the MFHS folder `cd MFHS` and run  `pip install -e.`  
4.Then install the packages `pip install -r reauireents.txt`

## 👉 Get Started  
First Download the dataset [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC) from SSL4MS.  
Then download the SAM2 pre-trained checkpoints named [sam2_hiera_large](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to obtain the corresponding checkpoint files.

## 👉 Training  
```shell
python train.py
```

## 👉 Testing  
```shell
python test.py
```
## 🤝 Acknowledgement
We are very grateful to the contributors of the following open-source projects and other referenced content.
[SAM2-UNet](https://github.com/WZH0120/SAM2-UNet)
[SSL4MIS](https://github.com/HiLab-git/SSL4MIS/tree/master)
[HyperYolo](https://github.com/iMoonLab/Hyper-YOLO)
[MCNet](https://github.com/ycwu1997/MC-Net)

