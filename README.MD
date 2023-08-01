# Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild

Official PyTorch implementation of Re-IQA, a No-Reference Image Quality Assessment algorithm proposed in [IEEE/CVF CVPR 2023](https://cvpr2023.thecvf.com/). Re-IQA achieves SoTA performance across popular NR-IQA databases : KonIQ, FLIVE, SPAQ, CLIVE, LIVE-IQA, CSIQ-IQA, TID-2013 and KADID.

Useful Links : [Preprint](https://arxiv.org/abs/2304.00451) [OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/papers/Saha_Re-IQA_Unsupervised_Learning_for_Image_Quality_Assessment_in_the_Wild_CVPR_2023_paper.pdf) [Supplementary](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Saha_Re-IQA_Unsupervised_Learning_CVPR_2023_supplemental.pdf) [YouTube Video](https://www.youtube.com/watch?v=gHIAC-L3eFg) [Slides](https://drive.google.com/file/d/1ckDpkJaj7Hk0KBX3g_0Kfpw3CBvPnGFE/view?usp=sharing) [Poster](https://drive.google.com/file/d/1aIob7YE77hT_LEARGftYdw1nINOLzENo/view?usp=sharing)

## Usage

The code has been tested on Linux systems with python 3.9, pytorch 1.13.1, torchvision0.14.1 and CUDA 11.7. Other dependencies can be installed via [requirements.txt](requirements.txt). 

## Training Re-IQA 

### Quality Aware Module

#### Download Training Data 

We follow the data training pipeline provided in [CONTRIQUE](https://github.com/pavancm/CONTRIQUE).
Create a directory ```mkdir training_data``` to store images used for training RE-IQA-Quality-Aware Module.
1. KADIS-700k : Download [KADIS-700k](http://database.mmsp-kn.de/kadid-10k-database.html) dataset. Store this data in the ```training_data/kadis700k``` directory. Note we do not use the synthetically distorted images as in CONTRIQUE.
2. AVA : Download [AVA](https://github.com/mtobeiyf/ava_downloader) dataset and store in the ```training_data/UGC_images/AVA_Dataset``` directory.
3. COCO : [COCO](https://cocodataset.org/#download) dataset contains 330k images spread across multiple competitions. We used 4 folders ```training_data/UGC_images/test2015, training_data/UGC_images/train2017, training_data/UGC_images/val2017, training_data/UGC_images/unlabeled2017``` for training.
4. CERTH-Blur : [Blur](https://mklab.iti.gr/results/certh-image-blur-dataset/) dataset images are stored in the ```training_data/UGC_images/blur_image``` directory. We only use the images labelled as "naturally-blurred" from the training (220) and evaluation (411) sets. We have renumbered the images in the provided [csv file](csv_files/moco_train.csv).
5. VOC : [VOC](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/) images are stored in the ```training_data/UGC_images/VOC2012``` directory.

#### Training Command


#### Obtaining Quality Aware Features

Download the quality-aware trained model from [Google Drive](https://drive.google.com/file/d/1DYMx8omn69yXUmBFL728JD3qMLNogFt8/view?usp=sharing) and store in a folder named ```re-iqa_ckpts```. Finally, to obtain Re-IQA Content-Aware features, run

```
python demo_quality_aware_feats.py --head mlp
```

### Content Aware Module 

We utilized the vanilla MoCo-v2 training code using ResNet-50 architecture and ImageNet database provided in the [PyContrast](https://github.com/HobbitLong/PyContrast) repository to train our Content Aware Module using the default settings. 

#### Obtaining Content Aware Features

Download the content-aware trained model from [Google Drive](https://drive.google.com/file/d/1TO-5fmZFT2_nt99j4IZen6vmXUb_UL3n/view?usp=sharing) and store in a folder named ```re-iqa_ckpts```. Finally, to obtain Re-IQA Content-Aware features, run

```
python demo_content_aware_feats.py --head mlp
```

### Training Linear Regressor




## Citation

If you use this code for your research, please cite the following paper:

[A. Saha, S. Mishra, and A. C. Bovik, “Re-IQA : Unsupervised Learning for Image Quality Assessment in the Wild,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023, https://doi.org/10.48550/arXiv.2304.00451.](https://arxiv.org/abs/2304.00451)

```
@InProceedings{Saha_2023_CVPR,
    author    = {Saha, Avinab and Mishra, Sandeep and Bovik, Alan C.},
    title     = {Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {5846-5855}
}
```

## Acknowledgements 

Our Code is based on [PyContrast](https://github.com/HobbitLong/PyContrast). We thank the [Yonglong Tian](https://github.com/HobbitLong) for making the code available.

## Contacts

- Avinab Saha ( avinab.saha@utexas.edu ) -- Graduate student, LIVE, Dept. of ECE, UT Austin.
- Sandeep Mishra ( sandy.mishra@utexas.edu ) -- Graduate student, LIVE, Dept. of ECE, UT Austin.