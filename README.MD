# Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild

Official PyTorch implementation of Re-IQA, a No-Reference Image Quality Assessment algorithm proposed in [IEEE/CVF CVPR 2023](https://cvpr2023.thecvf.com/). Re-IQA achieves SoTA performance across popular NR-IQA databases.

[Preprint](https://arxiv.org/abs/2304.00451) [OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/papers/Saha_Re-IQA_Unsupervised_Learning_for_Image_Quality_Assessment_in_the_Wild_CVPR_2023_paper.pdf) [Supplementary](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Saha_Re-IQA_Unsupervised_Learning_CVPR_2023_supplemental.pdf) [Video](https://www.youtube.com/watch?v=gHIAC-L3eFg) [Slides](https://drive.google.com/file/d/1ckDpkJaj7Hk0KBX3g_0Kfpw3CBvPnGFE/view?usp=sharing) [Poster](https://drive.google.com/file/d/1aIob7YE77hT_LEARGftYdw1nINOLzENo/view?usp=sharing)

## Usage

The code has been tested on Linux systems with python 3.9, pytorch 1.13 and CUDA 11.7. Please refer to [requirements.txt](requirements.txt) for installing dependent packages. 

## Training Re-IQA 

### Quality Aware Module

### Download Training Data 

We follow the data training pipeline provided in CONTRIQUE.

### Content Aware Module 

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

## Contacts

- Avinab Saha ( avinab.saha@utexas.edu ) -- Graduate student, LIVE, Dept. of ECE, UT Austin.
- Sandeep Mishra ( sandy.mishra@utexas.edu ) -- Graduate student, LIVE, Dept. of ECE, UT Austin.