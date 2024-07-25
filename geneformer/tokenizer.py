"""
Geneformer tokenizer.

Input data:
Required format: raw counts scRNAseq data without feature selection as .loom file
Required row (gene) attribute: "ensembl_id"; Ensembl ID for each gene
Required col (cell) attribute: "n_counts"; total read counts in that cell
Optional col (cell) attribute: "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria
Optional col (cell) attributes: any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below

Usage:
  from geneformer import TranscriptomeTokenizer
  tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ_major"}, nproc=4)
  tk.tokenize_data("loom_data_directory", "output_directory", "output_prefix")
"""

from __future__ import annotations
from typing import Literal
import pickle
from pathlib import Path

import logging

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import anndata as ad
import loompy as lp
import numpy as np
import scipy.sparse as sp
from datasets import Dataset

from time import time

import csv
import glob
import sys

logger = logging.getLogger(__name__)


# setting 
USE_GPU = "cuda:0"

# need file path
GENE_MEDIAN_FILE = "/path/to/mediam/dictionary/gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = "/path/to/token/dictionary/MLM-re_token_dictionary_v1.pkl"


def rank_genes(gene_vector, gene_tokens):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector)[:2048]

    return gene_tokens[sorted_indices]

def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]

    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])

def load_not_use_files(csv_file_path) :

    not_use_file_paths = []
    with open(csv_file_path, mode="r") as f :
        reader = csv.reader(f)
        for row in reader :
            not_use_file_paths.append(row[0])
    
    return not_use_file_paths



class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict=None,
        nproc=1,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : None, dict
            Dictionary of custom attributes to be added to the dataset.
            Keys are the names of the attributes in the loom file.
            Values are the names of the attributes in the dataset.
        nproc : int
            Number of processes to use for dataset mapping.
        gene_median_file : Path
            Path to pickle file containing dictionary of non-zero median
            gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl IDs:token).
        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)


        self.start_reading_file_num = 0

        # total cell nums
        self.learning_cell_nums = 22_446_161

        # final loom file flag
        self.last_dataset_flag = False

        # max cells in 1 dataset
        self.max_cells = 300_000

    def tokenize_data(
        self,
        data_directory: Path | str,
        output_directory: Path | str,
        output_prefix: str,
        file_format: Literal["loom", "h5ad"] = "loom",
        use_generator: bool = False,
    ):
        """
        Tokenize .loom files in loom_data_directory and save as tokenized .dataset in output_directory.

        Parameters
        ----------
        loom_data_directory : Path
            Path to directory containing loom files or anndata files
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        file_format : str
            Format of input files. Can be "loom" or "h5ad".
        use_generator : bool
            Whether to use generator or dict for tokenization.
        """

        data_set_num = self.start_reading_file_num
        while(1) :
            
            tokenized_cells, cell_metadata = self.tokenize_files(
                data_set_num, data_directory, file_format
            )
            if int(len(tokenized_cells)) == 0 :
                continue
            
            tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata, use_generator=use_generator)
            
            output_path = output_directory+"/"+output_prefix+"_"+str(data_set_num)+".dataset"
            tokenized_dataset.save_to_disk(output_path)
            print("saved to {}".format(output_path))
            
            if self.last_dataset_flag == True :
                break

            data_set_num += 1
        


    def tokenize_files(
        self, data_set_num, data_directory, file_format: Literal["loom", "h5ad"] = "loom"
    ):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        )

        total_loom_datas = len(glob.glob(data_directory+"*.loom"))
        total_cells_num = 0
        for enum1, file_path in enumerate(Path(data_directory).glob("*.{}".format(file_format))):
            if (enum1 < self.start_reading_file_num) : 
                continue
            file_found = 1
            print("=================================")
            print("[{} / {}]".format(enum1, total_loom_datas))
            print("Tokenizing : {}".format(file_path))
            
            file_tokenized_cells, file_cell_metadata, cells_num = tokenize_file_fn(file_path)
            tokenized_cells += file_tokenized_cells
            
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
            else:
                cell_metadata = None

            total_cells_num += cells_num


            if enum1 == total_loom_datas -1 :
                self.last_dataset_flag = True
            else :
                if total_cells_num >= self.max_cells :
                    self.start_reading_file_num = enum1 + 1
                    break
                else :
                    pass
        

        if file_found == 0:
            logger.error(
                f"No .{file_format} files found in directory {data_directory}.")
            raise

        return tokenized_cells, cell_metadata

    def tokenize_anndata(self, adata_file_path, target_sum=10_000, chunk_size=512):
        adata = ad.read(adata_file_path, backed="r")

        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in adata.var["ensembl_id"]]
        )[0]
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in adata.var["ensembl_id"][coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = adata.var["ensembl_id"][coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        try:
            _ = adata.obs["filter_pass"]
        except KeyError:
            var_exists = False
        else:
            var_exists = True

        if var_exists:
            filter_pass_loc = np.where(
                [i == 1 for i in adata.obs["filter_pass"]]
            )[0]
        elif not var_exists:
            print(
                f"{adata_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
            )
            filter_pass_loc = np.array([i for i in range(adata.shape[0])])

        tokenized_cells = []

        for i in range(0, len(filter_pass_loc), chunk_size):
            idx = filter_pass_loc[i:i+chunk_size]

            n_counts = adata[idx].obs['n_counts'].values[:, None]
            X_view = adata[idx, coding_miRNA_loc].X
            X_norm = (X_view / n_counts * target_sum / norm_factor_vector)
            X_norm = sp.csr_matrix(X_norm)

            tokenized_cells += [
                rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
                for i in range(X_norm.shape[0])
            ]

            # add custom attributes for subview to dict
            if self.custom_attr_name_dict is not None:
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += adata[idx].obs[k].tolist()
            else:
                file_cell_metadata = None

        return tokenized_cells, file_cell_metadata

    def tokenize_loom(self, loom_file_path, target_sum=10_000):
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys() 
            }
        
        loom_file_median = self.gene_median_file_path_dict[str(loom_file_path)]
        print(f"読み込んだloom fileのmedian file: {loom_file_median}")
        with open(loom_file_median, "rb") as f:
            self.gene_median_dict = pickle.load(f)
        
        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["row_attrs"]]
            )[0]
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["row_attrs"][coding_miRNA_loc]
                ]
            )
            coding_miRNA_ids = data.ra["row_attrs"][coding_miRNA_loc]
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )
            print(coding_miRNA_tokens.shape[0])

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            try:
                data.ca["filter_pass"]
            except AttributeError:
                var_exists = False
            else:
                var_exists = True

            if var_exists:
                filter_pass_loc = np.where(
                    [i == 1 for i in data.ca["filter_pass"]]
                )[0]
            elif not var_exists:
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                cells_counts = data.shape[1]
                filter_pass_loc = np.array([i for i in range(cells_counts)])

            # scan through .loom files and tokenize cells
            tokenized_cells = []
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                    subview[:, :]
                    / np.sum(subview[:, :], axis=0)
                    * target_sum
                    / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                if coding_miRNA_tokens.shape[0] > 0 :
                    tokenized_cells += [
                        tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens) # tokenize_cell(loomfileの1細胞の正規化発現量, loomfileのensembl_idのtoken_id)
                        for i in range(subview_norm_array.shape[1])
                    ]
                else :
                    pass

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                else:
                    file_cell_metadata = None

        return tokenized_cells, file_cell_metadata, cells_counts

    def create_dataset(self, tokenized_cells, cell_metadata, use_generator=False):
        print("Creating dataset.")
           
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None: # skip
            dataset_dict.update(cell_metadata)

        # create dataset
        if use_generator:
            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}
            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset        
        def truncate(example):
            example["input_ids"] = example["input_ids"][:2048]
            return example
        
        output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)
        

        # measure lengths of dataset
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example
        
        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=self.nproc
        )

        return output_dataset_truncated_w_length