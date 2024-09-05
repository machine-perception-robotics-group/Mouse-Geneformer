# mouse-Geneformer
Updating now...
Comming soon.

Writer: Keita Ito

## Abstract 
This repository contains the source code of mouse-Geneformer for analysing Single-cell RNA-sequence data of mouse. mosue-Geneformer is a model pre-trained on the large mouse single-cell dataset mouse-Genecorpus-20M and designed to map the mouse gene network. The mosue-Geneformer improves the accuracy of cell type classification of mouse cells and enables in silico perturbation experiments on mouse specimens.

## Citation
If you find this repository is useful. Please cite the following references.

```bibtex 
@article{aaaaaaa,
    author = {Keita Ito and Tsubasa Hirakawa and Shuji Shigenobu and Takayoshi Yamashita and Hironobu Fujiyoshi},
    title = {Mouse-Geneformer: A Deep Leaning Model for Mouse Single-Cell Transcriptome and Its Cross-Species Utility},
    journal = {xxxx},
    year = {20yy},
    pages = {zzzz-zzzz}
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
We have published the model files of mouse-Geneformer. `mouse-Geneformer` is base model. `mouse-Geneformer-12L-E20` is large model.

- [mouse-Geneformer](https://drive.google.com/...)
- [mouse-Geneformer-12L-E20](https://drive.google.com/...)