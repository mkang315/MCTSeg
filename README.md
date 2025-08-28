# Official MCTSeg
<div style="display:flex;justify-content: center">
<a href="https://github.com/mkang315/MCTSeg"><img src="https://img.shields.io/static/v1?label=GitHub&message=Code&color=black&logo=github"></a>
<a href="https://github.com/mkang315/MCTSeg"><img alt="Build" src="https://img.shields.io/github/stars/mkang315/MCTSeg"></a> 
<a href="https://huggingface.co/mkang315/MCTSeg"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=yellow"></a>
<a href="https://arxiv.org/abs/2404.14019"><img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2404.14019-b31b1b.svg"></a>
</div>

## Description
This is the source code for the paper, "A Multimodal Feature Distillation with CNN-Transformer Network for Brain Tumor Segmentation with Incomplete Modalities", of which I am the first author. The paper is a preprint without the intention to publish and available to download from [arXiv](https://arxiv.org/pdf/2404.14019).

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

## Referencing Guide
Please cite the paper if using this repository. Here is a guide to referencing this work in various styles for formatting your references:</br>

> Plain Text</br>
- **IEEE Reference Style**</br>
M. Kang, F. F. Ting, R. C.-W. Phan, Z. Ge, and C.-M. Ting, "A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities," 2024, *arXiv:2404.14019*.</br>

- **IEEE Full Name Reference Style**</br>
M. Kang, F. F. Ting, R. C.-W. Phan, Z. Ge, and C.-M. Ting, "A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities," arXiv:2404.14019, 2024.</br>
<sup>**NOTE:** This is a modification to the standard IEEE Reference Style and used by most IEEE/CVF conferences, including **CVPR**, **ICCV**, and **WACV**, to render first names in the bibliography as "Firstname Lastname" rather than "F. Lastname" or "Lastname, F.".</sup></br>
&nbsp;- **IJCAI Full Name-Year Variation**</br>
\[Kang *et al.*, 2024\] Ming Kang, Fung Fung Ting, Raphaël C.-W. Phan, Zongyuan Ge, and Chee-Ming Ting. A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities. *arXiv preprint arXiv:2404.14019*, 2024.</br>
&nbsp;- **ACL Full Name-Year Variation**</br>
Ming Kang, Fung Fung Ting, Raphaël C.-W. Phan, Zongyuan Ge, and Chee-Ming Ting. 2024. A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities. *arXiv preprint arXiv:2404.14019*.</br>

- **Nature Reference Style**</br>
Kang, M., Ting, F. F., Phan, R. C.-W., Ge, Z. & Ting, C.-M. A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities. Preprint at https://arxiv.org/abs/2404.14019 (2024).</br>

- **Springer Reference Style**</br>
Kang, M., Ting, F.F., Phan, R.C.-W., Ge, Z., Ting, C.-M.: A multimodal feature distillation with CNN-transformer network for brain tumor segmentation with incomplete modalities (2024), arXiv preprint [arXiv:2404.14019](https://arxiv.org/abs/2404.14019)</br>

- **Elsevier Numbered Style**</br>
M. Kang, F.F. Ting, R.C.-W. Phan, Z. Ge, C.-M. Ting, A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities, arXiv preprint arXiv:2404.14019 (2024).</br>

- **Elsevier Name–Date (Harvard) Style**</br>
Kang, M., Ting, F.F., Phan, R.C.-W., Ge, Z., Ting, C.-M., 2024. A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities. arXiv preprint arXiv:2404.14019.</br>

- **Elsevier Vancouver Style**</br>
Kang M, Ting FF, Phan RC-W, Ge Z, Ting C-M. A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities. ArXiv \[Preprint\]. 2024 arXiv:2404.14019 \[posted 2024 Apr 22\]: \[10 p.\]. Available from: https://arxiv.org/abs/2404.14019 doi: https://doi.org/10.48550/arXiv.2404.14019</br>

- **Elsevier Embellished Vancouver Style**</br>
Kang M, Ting FF, Phan RC-W, Ge Z, Ting C-M. A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities. ArXiv \[Preprint\]. 2024 arXiv:2404.14019 \[posted 2024 Apr 22\]: \[10 p.\]. Available from: https://arxiv.org/abs/2404.14019 doi: https://doi.org/10.48550/arXiv.2404.14019</br>

- **APA7 (Author–Date) Style**</br>
Kang, M., Ting, F. F., Phan, R. C.-W., Ge, Z., & Ting, C.-M. (2024). *A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities*. ArXiv, https://doi.org/10.48550/arXiv.2404.14019</br>
&nbsp;- **ICML (Author–Year) Variation**</br>
Kang, M., Ting, F. F., Phan, R. C.-W., Ge, Z., and Ting, C.-M. A multimodal feature distillation with CNN-Transformer network for brain tumor segmentation with incomplete modalities. *arXiv preprint arXiv:2404.14019*, 2024.</br>
<sup>**NOTE:** For **NeurIPS** and **ICLR**, any reference/citation style is acceptable as long as it is used consistently. The sample of references in Formatting Instructions For NeurIPS almost follows APA7 (author–date) style and that in Formatting Instructions For ICLR Conference Submissions is similar to IJCAI full name-year variation.</sup>

> BibTeX Format</br>
```
\begin{thebibliography}{1}
\bibitem{bib1} M. Kang, R. C.-W. Phan, F. F. Ting, Z. Ge, and C.-M. Ting, ``A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities,'' 2024, {\em arXiv:2406.08634}.
\end{thebibliography}
```
```
@unpublished{Kang24Mctseg,
  author = "Kang, Ming and Ting, Fung Fung and Phan, Rapha{\"e}l C.-W. and Ge, Zongyuan and Ting, Chee-Ming",
  title = "A multimodal feature distillation with {CNN}-{T}ransformer network for brain tumor segmentation with incomplete modalities",
  howpublished = "arXiv preprint",
  year = "2024",
  doi= "10.48550/arXiv.2404.14019",
  url = "https://doi.org/10.48550/arXiv.2404.14019"
}
```
```
@aunpublished{Kang24Mctseg,
  author = "Ming Kang and Fung Fung Ting and Rapha{\"e}l C.-W. Phan and Zongyuan Ge and Chee-Ming Ting",
  title = "A multimodal feature distillation with cnn-transformer network for brain tumor segmentation with incomplete modalities",
  howpublished = "arXiv preprint",
  note = "arXiv:2404.14019",
  year = "2024",
}
```
<sup>**NOTE:** Please remove some optional *BibTeX* fields/tags such as `series`, `volume`, `address`, `url`, and so on if the *LaTeX* compiler produces an error. Author names may be manually modified if not automatically abbreviated by the compiler under the control of the bibliography/reference style (i.e., .bst) file. The *BibTex* citation key may be `bib1`, `b1`, or `ref1` when references appear in numbered style in which they are cited. The quotation mark pair `""` in the field could be replaced by the brace `{}`, whereas the brace `{}` in the *BibTeX* field/tag `title` plays a role of keeping letters/characters/text original lower/uppercases or sentence/capitalized cases unchanged while using Springer Nature bibliography style files, for example, sn-nature.bst.</sup>

## License
MCTSeg is released under the BSD 3-Clause "New" or "Revised" License. Please see the [LICENSE](https://github.com/mkang315/MCTSeg/blob/main/LICENSE) file for more information.

## Copyright Notice
Many utility codes of our project base on the codes of [RFNet](https://github.com/dyh127/RFNet), [M<sup>3</sup>AE](https://github.com/ccarliu/m3ae), and [Vision Transformer PyTorch](https://github.com/asyml/vision-transformer-pytorch) repositories.
