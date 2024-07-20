# Official MCTSeg
This is the source code for the paper, "A Multimodal Feature Distillation with CNN-Transformer Network for Brain Tumor Segmentation with Incomplete Modalities", of which I am the first author.<!-- [IEEE Journal of Biomedical and Health Informatics (IEEE J-BHI or IEEE JBHI)](https://www.embs.org/jbhi)-->

## Model
The Multimodal feature distillation with CNN-Transformer hybrid network for incomplete multimodal brain tumor Segmentation (MCTSeg) model configuration (i.e., network construction) file is net.py in the directory [.\model](https://github.com/mkang315/MCTSeg/tree/main/model).
To train and test by running [train.py](https://github.com/mkang315/MCTSeg/blob/main/train.py) and [test.py](https://github.com/mkang315/MCTSeg/blob/main/test.py).

Recommended dependencies:
```
Python <= 3.8
Torch <= 1.7.1
CUDA <= 11.1
```

## Evaluation
Datasets Brain Tumor Segmentation (BraTS) Challenge 2018/2020 ([BraTS2018](https://www.med.upenn.edu/sbia/brats2018.html)/[BraTS2020](https://www.med.upenn.edu/cbica/brats2020/)).

## Suggested Citation
Our manuscript has been uploaded on [arXiv](https://arxiv.org/abs/2404.14019). Please cite our paper if you use code from this repository:
> Plain Text

- IEEE Style</br>
M. Kang, F. F. Ting, R. C.-W. Phan, Z. Ge, and C.-M. Ting, "A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities," arXiv:2404.14019, Apr. 2024.</br>

- Nature Style</br>
Kang, M., Ting, F. F., Phan, R. C.-W., Ge, Z., & Ting, C.-M.. A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities. Preprint at https://arxiv.org/abs/2404.14019 (2024).</br>

- Springer Style</br>
Kang, M., Ting, F. F., Phan, R.C.-W., Ge, Z., Ting, C.-M.: A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities. arXiv preprint [arXiv:2404.14019](https://arxiv.org/abs/2404.14019) (2024)</br>

- Elsevier Reference Style</br>
M. Kang, F.F. Ting, R.C.-W. Phan, Z. Ge, C.-M. Ting, A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities, arXiv:2404.14019, 2024.</br>

<!--
> BibTeX Format</br>
```
\begin{thebibliography}{1}
\bibitem{bib1} M. Kang, R. C.-W. Phan, F. F. Ting, Z. Ge, and C.-M. Ting, "A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities," {\it IEEE J. Biomed. Health Inform.}, in press, 105057, May 2024.
\end{thebibliography}
```
```
@article{Kang24Mctseg,
  author = "Kang, Ming and Ting, Fung Fung and Phan, Rapha{\"e}l C.-W. and Ge, Zongyuan and Ting, Chee-Ming",
  title = "A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities",
  journal = "IEEE J. Biomed. Health Inform.",
  volume = "28",
  number = "00",
  pages = "0000--0000",
  publisher = "IEEE",
  address = "Piscataway",
  year = "2024",
  doi= "10.1109/JBHI.2024.3400000",
  url = "https://doi.org/10.1109/JBHI.2024.3400000"
}
```
```
@article{Kang24Mctseg,
  author = "Ming Kang and Fung Fung Ting and Rapha{\"e}l C.-W. Phan and Zongyuan Ge and Chee-Ming Ting",
  title = "A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities",
  journal = "IEEE J. Biomed. Health Inform.",
  volume = "28",
  number = "00",
  pages = "0000--0000",
  note = "in press",
  month = "Dec.",
  year = "2024",
}
```
<sup>**NOTE:** Please remove some optional *BibTeX* fields, for example, `series`, `volume`, `address`, `url` and so on, while the *LaTeX* compiler produces an error. Author names may be manually modified if not automatically abbreviated by the compiler under the control of the .bst file. `kang2023rcsyolo` could be `b1`, `bib1`, or `ref1` when references appear in the order in which they are cited. The quotation mark pair `""` in the field could be replaced by the brace `{}`. </sup>
-->

## License
MCTSeg is released under the BSD 3-Clause "New" or "Revised" License. Please see the [LICENSE](https://github.com/mkang315/PKGSeg/blob/main/LICENSE) file for more information.

## Copyright Notice
Many utility codes of our project base on the codes of [PyTorch-3DUNet](https://github.com/wolny/pytorch-3dunet), [mmFormer](https://github.com/YaoZhang93/mmFormer), [Vision Transformer PyTorch](https://github.com/asyml/vision-transformer-pytorch), and [Factor-Transfer-pytorch](https://github.com/Jangho-Kim/Factor-Transfer-pytorch) repositories.
