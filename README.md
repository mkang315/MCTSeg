# Official MFDSeg
This is the source code for the paper, "Multimodal Feature Distillation Based CNN-Transformer Hybrid Networks for MRI Brain Tumor Segmentation with Incomplete Modalities", of which I am the first author.

## Model
The model configuration (i.e., network construction) file is net.py in the directory .\model.
To train and test by running train.py and test.py.

## Evaluation
Datasets Brain Tumor Segmentation (BraTS) Challenge 2018/2020 ([BraTS2018](https://www.med.upenn.edu/sbia/brats2018.html)/[BraTS2020](https://www.med.upenn.edu/cbica/brats2020/)).

## Suggested Citation
Our manuscript has been uploaded on [arXiv](https://arxiv.org/abs/2309.12585). Please cite our paper if you use code from this repository:
> Plain Text

- *IEEE* Style</br>
M. Kang, C.-M. Ting, F. F. Ting, and R. C.-W. Phan, "BGF-YOLO: Enhanced yolov8 with multiscale attentional feature fusion for brain tumor detection," arXiv:2309.12585 [cs.CV], Jun. 2023.</br>

- *Nature* Style</br>
Kang, M., Ting, C.-M., Ting, F. F. & Phan, R. C.-W. BGF-YOLO: enhanced YOLOv8 with multiscale attentional feature fusion for brain tumor detection. Preprint at https://arxiv.org/abs/2309.12585 (2023).</br>

- *Springer* Style</br>
Kang, M., Ting, C.-M., Ting, F. F., Phan, R.C.-W.: BGF-YOLO: enhanced YOLOv8 with multiscale attentional feature fusion for brain tumor detection. arXiv preprint [arXiv:2309.12585](https://arxiv.org/abs/2309.12585) (2023)</br>

## License
PKGSeg is released under the BSD 3-Clause "New" or "Revised" License. Please see the [LICENSE](https://github.com/mkang315/PKGSeg/blob/main/LICENSE) file for more information.

## Copyright Notice
Many utility codes of our project base on the codes of [PyTorch-3DUNet](https://github.com/wolny/pytorch-3dunet), [mmFormer](https://github.com/YaoZhang93/mmFormer), [Vision Transformer PyTorch](https://github.com/asyml/vision-transformer-pytorch), and [Factor-Transfer-pytorch](https://github.com/Jangho-Kim/Factor-Transfer-pytorch) repositories.
