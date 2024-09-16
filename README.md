# mouse-Geneformer

Writer: Keita Ito

## Abstract 
This repository contains the source code of mouse-Geneformer for analysing Single-cell RNA-sequence data of mouse. mosue-Geneformer is a model pre-trained on the large mouse single-cell dataset mouse-Genecorpus-20M and designed to map the mouse gene network. The mosue-Geneformer improves the accuracy of cell type classification of mouse cells and enables in silico perturbation experiments on mouse specimens.

<!-- PLOS Genetics の URL を載せる-->
<!-- [[PLOS Genetics](hhttps://journals.plos.org/plosgenetics/...)]-->

<!--bioRxiv の URL を載せる -->
[[bioRxiv](https://www.biorxiv.org/content/10.1101/2024.09.09.611960v1)]

## Citation
If you find this repository is useful. Please cite the following references.

<!-- PLOS Genetics の bibtex を載せる -->

<!--bioRxiv の bibtex を載せる -->
```bibtex 
@article{Shigenobu2024.09.09.611960,
	author = {Shigenobu, Shuji and Ito, Keita and Hirakawa, Tsubasa and Fujiyoshi, Hironobu and Yamashita, Takayoshi},
    title = {Mouse-Geneformer: A Deep Leaning Model for Mouse Single-Cell Transcriptome and Its Cross-Species Utility},
    journal = {bioRxiv},
    year = {2024},
    URL = {https://www.biorxiv.org/content/early/2024/09/13/2024.09.09.611960}
}
```
## Enviroment
Our source code is based on mplemented with PyTorch. 
Required PyTorch and Python version is as follows:
- PyTorch : 2.0.1
- Python vision : 3.8.10

## Execute
Example of Pretraining run command is as follows:

#### Pretraining
```bash
# mosue-Genecorpus-20M dataset
./start_pretrain_geneformer.sh
```
Downstream task is executed in jupyter files (`cell_classification.ipynb` and `in_silico_perturbation.ipynb`)

## Trained model
We have published the model files of mouse-Geneformer.

 `mouse-Geneformer` is base model. `mouse-Geneformer-12L-E20` is large model.

<!-- mouse-Geneformer の事前学習済みモデルの google drive の URL を載せる-->
- [mouse-Geneformer](https://drive.google.com/file/d/1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg/view?usp=sharing)
<!-- mouse-Geneformer-L12-E20 の事前学習済みモデルの google drive の URL を載せる-->
- [mouse-Geneformer-12L-E20](https://drive.google.com/file/d/1xKMyFA4JJeRigcJPsU2XNyxEW25Q247u/view?usp=sharing)