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

#GENE_MEDIAN_FILE = Path(__file__).parent / "gene_median_dictionary.pkl"
#TOKEN_DICTIONARY_FILE = Path(__file__).parent / "token_dictionary.pkl"

# setting 
ORGANISM = "mouse"
MH_FLAG = False
USE_GPU = "cuda:7"

GENE_MEDIAN_FILE = "/mnt/keita/data/scRNA-datas/mouse_data/mouse-genecorpus-20M/data1/tokens/gene_median_dictionary_0-3000_0-001.pkl"

if ORGANISM == "human" :
    TOKEN_DICTIONARY_FILE = "/mnt/keita/data/scRNA-datas/human_data/Geneformer/genecorpus-30M/token_dictionary_human_myocardial-covid19-ctchuman_mouse_cop1ko-easy-hard.pkl"
elif ORGANISM == "mouse" :
    TOKEN_DICTIONARY_FILE = "/mnt/keita/data/scRNA-datas/mouse_data/mouse-Geneformer/mouse-genecorpus-20M/data1-v2/tokens/MLM-re_token_dictionary_v1_add_niigata.pkl"
else :
    print("in tokenizer.py: Not select ORGANISM (human or mouse)")
    sys.exit(1)
    
def rank_genes(gene_vector, gene_tokens):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    # loomfileの細胞のゼロでない正規化発現量を大きい順にソートしてる
    sorted_indices = np.argsort(-gene_vector)[:2048]
    # loomfileのensembl_idのtoken_idを正規化発現量が大きい順にしてる(np.array)

    return gene_tokens[sorted_indices]
    #return np.array([int(gene_tokens[sorted_indices[i]]) for i in range(gene_tokens[sorted_indices].shape[0])])


def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    # loomfileの細胞のゼロでない正規化発現量のindexを持ってきている
    nonzero_mask = np.nonzero(gene_vector)[0]
    # rank by median-scaled gene values
    # loomfileがrank value encodingされたトークンデータを返してる(np.array)
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
        nproc=1, # 4にしたほうがいい?
        dataset_num=-1,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : None, dict
            Dictionary of custom attributes to be added to the dataset.    データセットに追加するカスタム属性の辞書。
            Keys are the names of the attributes in the loom file.         キーはloomファイルの属性名。
            Values are the names of the attributes in the dataset.         値はデータセットの属性名。
        nproc : int
            Number of processes to use for dataset mapping.
        gene_median_file : Path
            Path to pickle file containing dictionary of non-zero median
            gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl IDs:token).
        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        # keyがloomfileのcolumns名(細胞名)でvalueがトークン化するcolumns名(細胞名)となる辞書(その辞書があれば辞書を格納するが，なければNoneを格納)
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        # 非ゼロ中央値ファイルの中身を保存
        #with open(gene_median_file, "rb") as f:
        #    self.gene_median_dict = pickle.load(f)
        
        with open(gene_median_file, "rb") as f:
            self.gene_median_file_path_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        # トークンファイルの中身を保存
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary
        # 非ゼロ中央値のEnsembl ID数+2(特殊文字2個)を格納
        #self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        # loomfileの細胞をトークン化するために，特定のEnsembl ID行を取得するための辞書作成
        #self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

        # トークン化するために読み始めるloomfileのインデックス番号
        # defaultの値は-1
        # self.start_reading_file_num = -1    : はじめのファイルから読む
        # self.start_reading_file_num = (1~x) : (1~x)のファイルから読む
        self.start_reading_file_num = -1

        # 読み終えたファイルの個数
        self.end_file_nums = 0

        # appach arrow形式のトークンのデータセット数(1つのdatasetfileにメモリの関係でトークンデータを保存できないときに使用)
        # 1つのdatasetfileに保存する細胞数は「学習させる総細胞数 / self.dataset_nums」個の細胞となる
        # defaultの値は1
        self.dataset_nums = dataset_num

        # 学習させる総細胞数
        self.learning_cell_nums = 22_446_161

        # 最後に作成するデータセットかどうかを判断するフラグ
        self.last_dataset_flag = False

        # 1datasetに含む細胞数の最大値
        self.max_cells_of_one_dataset = 300_000

        self.starting_program_time = time()

        print("=================================")
        print("Setting")
        print("TranscriptomeTokenizer")
        print("custom_attr_name_dict = {}".format(self.custom_attr_name_dict))
        print("gene_median_file = {}".format(gene_median_file))
        print("token_dictionary_file = {}".format(token_dictionary_file))
        print("nproc = {}".format(self.nproc))
        print("dataset_nums = {}".format(self.dataset_nums))
        print("learning_cell_nums = {}".format(self.learning_cell_nums))
        print("planned_dataset_nums = {}".format(int(self.learning_cell_nums / self.dataset_nums)+1))

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
            Path to directory containing loom files or anndata files                loom ファイルまたは anndata ファイルを含むディレクトリへのパス。
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset        .dataset形式のファイルを保存するパス
        output_prefix : str
            Prefix for output .dataset                                               出力する.datasetの接頭辞
        file_format : str
            Format of input files. Can be "loom" or "h5ad".                          入力ファイルのフォーマット。loom "または "h5ad"。
        use_generator : bool
            Whether to use generator or dict for tokenization.                       トークン化にジェネレーターを使うかdictを使うか
        """

        for data_set_num in range(0+1,150+1,1) :

            if data_set_num < 0 :
                continue
            
            # ランク値エンコーディングを実行し，エンコーディング結果と細胞のメタデータを格納
            # ==== 戻り値 ====
            # tokenized_cells：全loomfileの全細胞のトークン(list型[入れ子])
            # cell_metadata：loomfileの細胞に関するメタデータ．self.custom_attr_name_dict = Noneなため，file_cell_metadata = Noneとなる．
            tokenized_cells, cell_metadata = self.tokenize_files(
                data_set_num, data_directory, file_format
            )
            if int(len(tokenized_cells)) == 0 :
                continue
            
            # エンコーディング結果と細胞のメタデータからarrow形式のトークン化したデータセットを作成
            # == 戻り値 ==
            # datasetの各細胞のトークン数が2048以下となり，dataset全体の大きさを追加したappach arrow形式のdataset
            tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata, use_generator=use_generator)
            
            # 作成したappach arrow形式のトークン化したデータセットを保存するパスを作成
            #output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
            output_path = output_directory+"/"+output_prefix+"_"+str(data_set_num)+".dataset"
            # パスに作成したarrow形式のトークン化したデータセットを保存
            tokenized_dataset.save_to_disk(output_path)
            print("{}に保存完了".format(output_path))
            
            """
            if (((self.dataset_nums - 2) < data_set_num) and (data_set_num < self.dataset_nums)) :
                self.last_dataset_flag = True
            """

            if self.last_dataset_flag == True :
                break


    def tokenize_files(
        self, data_set_num, data_directory, file_format: Literal["loom", "h5ad"] = "loom"
    ):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            # loomfileのcolumns名(細胞名)をリストに格納
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            # トークン化するcolumns名(細胞名)をkeyとして[]をvalueとした辞書型を生成（初期化）
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        # loops through directories to tokenize .loom files
        # loomfileが存在するかのフラグ 0は存在しない．1は存在する．
        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        # loomfileのtokenizer関数を選択し，格納する
        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        )

        # loomfile全部をfor-loopで参照
        total_loom_datas = len(glob.glob(data_directory+"*.loom"))
        start_time1 = time()
        total_cells_num = 0
        loaded_loomfile_num = 0
        for enum1, file_path in enumerate(Path(data_directory).glob("*.{}".format(file_format))): # file_format = loom format
        # for enum1, file_path in enumerate()
            if (enum1 < self.start_reading_file_num) : 
                continue
            # loomfileが存在するかのフラグ fileは存在する．
            file_found = 1
            start_time2 = time()
            print("=================================")
            print("[{} / {}]".format(enum1, total_loom_datas))
            print("Tokenizing : {}".format(file_path))
            
            # tokenize_loom関数を実行
            # ==== 戻り値 ====
            # file_tokenized_cells：loomfile内の細胞の正規化発現量が大きい順になったEnsembl IDのトークンIDが格納されている．
            # file_cell_metadata：loomfileの細胞に関するメタデータ．self.custom_attr_name_dict = Noneなため，file_cell_metadata = Noneとなる．
            file_tokenized_cells, file_cell_metadata, cells_num, not_use_file_name = tokenize_file_fn(file_path) #tokenize_loom()
            # 全loomfileの細胞のトークンIDを格納
            tokenized_cells += file_tokenized_cells
            #print("====")
            #print(tokenized_cells)
            #print(type(tokenized_cells))
            #print("====")

            """
            if not_use_file_name is not None :
                with open("/mnt/keita/data/scRNA-data/normal_mouse/mouse_datas/tmp/not_use_tokenized_cells_file.csv", mode="a") as f:
                    f.write(not_use_file_name+"\n")
            """


            # loomfileの細胞に対してのmetadataを格納．self.custom_attr_name_dict = Noneなため，cell_metadata = None
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
            else:
                cell_metadata = None

            loaded_loomfile_num += 1
            total_cells_num += cells_num
            
            #print("1 dataset in {} loom files".format(int((total_loom_datas - self.end_file_nums) / (self.dataset_nums - 17))))
            print("current total cells : {}".format(total_cells_num))
            print("Number of loaded loom files : {} ".format(loaded_loomfile_num))
            print(f"{file_path} Tokenizing time[s]: {time()-start_time2}")
            print(f"Time token to create 1 dataset[s]: {time()-start_time1}")
            print(f"Time toking to create {self.dataset_nums} dataset[s]: {time()-self.starting_program_time}")


            """
            if (1 < self.dataset_nums) and (self.last_dataset_flag == False):
                if ((((int((total_loom_datas - self.end_file_nums) / (self.dataset_nums - 0)) * (data_set_num - 0)) + self.end_file_nums) - 2) < enum1) :
                #if (((int(total_loom_datas / self.dataset_nums) * data_set_num) - 2) < enum1) :
                    print("{}のうち1つのデータセットを作成".format(self.dataset_nums))
                    self.start_reading_file_num = enum1 + 1
                    break
                else :
                    pass
            else :
                pass
            """

            if enum1 == total_loom_datas-1 :
                print("{}番目のデータセットは最後のデータセット".format(enum1))
                self.last_dataset_flag = True
                break
            else :
                if total_cells_num > self.max_cells_of_one_dataset :
                    print("{}から{}まで（計{}個）のファイルでデータセットを作成".format(self.start_reading_file_num, enum1, loaded_loomfile_num))
                    self.start_reading_file_num = enum1 + 1
                    print("次に読み込むファイルは{}番目のファイル".format(self.start_reading_file_num))
                    break
                else :
                    pass
            
            
            

            """
            # テスト
            if enum1 > 4 :
                break
            """



        # fileが存在しなかった場合の処理
        if file_found == 0:
            logger.error(
                f"No .{file_format} files found in directory {data_directory}.")
            raise
        
        # tokenized_cells：全loomfileの全細胞のトークン
        # loomfileの細胞に対してのmetadataを格納．self.custom_attr_name_dict = Noneなため，cell_metadata = None
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
            # loomfileのcolumns名(細胞名)をkeyとして[]をvalueとした辞書型を生成（初期化）
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys() 
            }
        
        file_name = None

        # loom fileにあった中央値ファイルをロードする
        loom_file_median = self.gene_median_file_path_dict[str(loom_file_path)]
        print(f"読み込んだloom fileのmedian file: {loom_file_median}")
        with open(loom_file_median, "rb") as f:
            self.gene_median_dict = pickle.load(f)
        
        # gene keys for full vocabulary
        # 非ゼロ中央値のEnsembl ID数+2(特殊文字2個)を格納
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        # loomfileの細胞をトークン化するために，特定のEnsembl ID行を取得するための辞書作成
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            # loomfileのensembl_idのindexをとってきている
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["row_attrs"]] # "ensembl_id"を"row_attrs"に変更する必要がある
            )[0]
            # loomfileのensembl_idの中央値をとってきている
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["row_attrs"][coding_miRNA_loc]
                ]
            )
            # loomfileのensembl_idを格納している
            coding_miRNA_ids = data.ra["row_attrs"][coding_miRNA_loc]
            # loomfileのensembl_idをkeysとしたときのvalueであるtoken_idをとってきている
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )
            print(coding_miRNA_tokens.shape[0])
                
            #print([type(coding_miRNA_tokens[i]) for i in range(coding_miRNA_tokens.shape[0])])

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            # data.ca["filter_pass"]は存在しないため，exceptが実行され，var_exists = Falseとなる
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
                # loomfileの全ての細胞がトークンか対象となる
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                cells_counts = data.shape[1]
                filter_pass_loc = np.array([i for i in range(cells_counts)])

            # scan through .loom files and tokenize cells
            # loomfileをスキャンし，loomfile内の細胞をトークン化する
            tokenized_cells = []
            # itemsに格納された条件に当う列（細胞）を全て取得．故に，for-loopは1回である．
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                # フィルタリング後の細胞のensembl_idがある遺伝子をとってきている
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                # ランク値エンコーディング：正規化発現量の計算
                # 式：(subview[:,:] * (1 / np.sum(subview, axis=1)) * 10000 * (1 / norm_factor_vector[:, None]))
                subview_norm_array = (
                    subview[:, :]                  # loomfile内の細胞の遺伝子発現量
                    / np.sum(subview[:, :], axis=0)#subview.ca.n_counts          # loomfile内の細胞の遺伝子発現総数で正規化
                    * target_sum                   # 10000倍
                    / norm_factor_vector[:, None]  # loomfileにあるEnsembl ID（遺伝子）の中央値で正規化
                )
                # tokenize subview gene vectors
                # ランク値エンコーディング：ランク値に基づいて大きい順にした遺伝子をトークン化
                if coding_miRNA_tokens.shape[0] > 0 :
                    tokenized_cells += [
                        tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens) # tokenize_cell(loomfileの1細胞の正規化発現量, loomfileのensembl_idのtoken_id)
                        for i in range(subview_norm_array.shape[1])
                    ]
                else :
                    file_name = str(loom_file_path)

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    # loomfileのcolumns名(細胞名)のvalueにフィルタリング後の細胞のensembl_idがある遺伝子をもってきてる(トークン化する細胞を持ってきている)
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                else:
                    file_cell_metadata = None
        
        # tokenized_cells：loomfile内の細胞の正規化発現量が大きい順になったEnsembl IDのトークンIDが格納されている．
        # file_cell_metadata：loomfileの細胞に関するメタデータ．self.custom_attr_name_dict = Noneなため，file_cell_metadata = Noneとなる．

        return tokenized_cells, file_cell_metadata, cells_counts, file_name

    def create_dataset(self, tokenized_cells, cell_metadata, use_generator=False):
        print("Creating dataset.")
        # create dict for dataset creation
        # トークンを辞書型にして格納
        # dataset_dict：keyに"input_ids"，valueに全loomfileの全細胞のトークン(list)を持つ辞書型
        
        """
        print("====")
        print(len(tokenized_cells))
        print(type(tokenized_cells))
        print("====")
        """


        # 外部ファイルに保存してあるtokenized_cellsに格納している全loomfileの細胞のトークンIDを取得(list型)
        # tokenized_cellsを取得するときは，tokenized_cellsのデータ構造のまま取得    
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None: # skip
            dataset_dict.update(cell_metadata)

        # create dataset
        if use_generator: # use_generator = Falseなため，skip
            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}
            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            # appach arrow形式の.datasetにdataset_dictを保存
            #output_dataset = Dataset.from_dict(dataset_dict)
            output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset
        # 各細胞のトークンが2048以下のトークン数になるように整形
        
        def truncate(example):
            example["input_ids"] = example["input_ids"][:2048]
            return example
        
        
        # datasetの各細胞のトークンが2048以下になったものを追加 or 保存
        output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)
        

        # measure lengths of dataset
        # dataset全体の大きさを計算
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example
        
        # dataset全体の大きさを計算し，datasetに追加 or 保存
        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=self.nproc
        )

        # datasetの各細胞のトークンが2048以下となり，dataset全体の大きさを追加したdatasetをreturn
        return output_dataset_truncated_w_length
    
    
    
class In_Silico_TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict=None,
        nproc=1, # 4にしたほうがいい?
        dataset_num=-1,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : None, dict
            Dictionary of custom attributes to be added to the dataset.    データセットに追加するカスタム属性の辞書。
            Keys are the names of the attributes in the loom file.         キーはloomファイルの属性名。
            Values are the names of the attributes in the dataset.         値はデータセットの属性名。
        nproc : int
            Number of processes to use for dataset mapping.
        gene_median_file : Path
            Path to pickle file containing dictionary of non-zero median
            gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl IDs:token).
        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        # keyがloomfileのcolumns名(細胞名)でvalueがトークン化するcolumns名(細胞名)となる辞書(その辞書があれば辞書を格納するが，なければNoneを格納)
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        # 非ゼロ中央値ファイルの中身を保存
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        # トークンファイルの中身を保存
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary
        # 非ゼロ中央値のEnsembl ID数+2(特殊文字2個)を格納
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        # loomfileの細胞をトークン化するために，特定のEnsembl ID行を取得するための辞書作成
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

        # トークン化するために読み始めるloomfileのインデックス番号
        # defaultの値は-1
        # self.start_reading_file_num = -1    : はじめのファイルから読む
        # self.start_reading_file_num = (1~x) : (1~x)のファイルから読む
        self.start_reading_file_num = -1

        # 読み終えたファイルの個数
        self.end_file_nums = 0

        # appach arrow形式のトークンのデータセット数(1つのdatasetfileにメモリの関係でトークンデータを保存できないときに使用)
        # 1つのdatasetfileに保存する細胞数は「学習させる総細胞数 / self.dataset_nums」個の細胞となる
        # defaultの値は1
        self.dataset_nums = dataset_num

        # 学習させる総細胞数
        self.learning_cell_nums = 22_446_161

        # 最後に作成するデータセットかどうかを判断するフラグ
        self.last_dataset_flag = False

        # 1datasetに含む細胞数の最大値
        self.max_cells_of_one_dataset = 300_000

        self.starting_program_time = time()

        print("=================================")
        print("Setting")
        print("TranscriptomeTokenizer")
        print("custom_attr_name_dict = {}".format(self.custom_attr_name_dict))
        print("gene_median_file = {}".format(gene_median_file))
        print("token_dictionary_file = {}".format(token_dictionary_file))
        print("nproc = {}".format(self.nproc))
        print("dataset_nums = {}".format(self.dataset_nums))
        print("learning_cell_nums = {}".format(self.learning_cell_nums))
        print("planned_dataset_nums = {}".format(int(self.learning_cell_nums / self.dataset_nums)+1))

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
            Path to directory containing loom files or anndata files                loom ファイルまたは anndata ファイルを含むディレクトリへのパス。
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset        .dataset形式のファイルを保存するパス
        output_prefix : str
            Prefix for output .dataset                                               出力する.datasetの接頭辞
        file_format : str
            Format of input files. Can be "loom" or "h5ad".                          入力ファイルのフォーマット。loom "または "h5ad"。
        use_generator : bool
            Whether to use generator or dict for tokenization.                       トークン化にジェネレーターを使うかdictを使うか
        """

        for data_set_num in range(0+1,84+1,1) :

            if data_set_num < 0 :
                continue
            
            # ランク値エンコーディングを実行し，エンコーディング結果と細胞のメタデータを格納
            # ==== 戻り値 ====
            # tokenized_cells：全loomfileの全細胞のトークン(list型[入れ子])
            # cell_metadata：loomfileの細胞に関するメタデータ．self.custom_attr_name_dict = Noneなため，file_cell_metadata = Noneとなる．
            tokenized_cells, cell_metadata = self.tokenize_files(
                data_set_num, data_directory, file_format
            )
            if int(len(tokenized_cells)) == 0 :
                continue
            
            # エンコーディング結果と細胞のメタデータからarrow形式のトークン化したデータセットを作成
            # == 戻り値 ==
            # datasetの各細胞のトークン数が2048以下となり，dataset全体の大きさを追加したappach arrow形式のdataset
            tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata, use_generator=use_generator)
            
            # 作成したappach arrow形式のトークン化したデータセットを保存するパスを作成
            #output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
            output_path = output_directory+"/"+output_prefix+"_"+str(data_set_num)+".dataset"
            # パスに作成したarrow形式のトークン化したデータセットを保存
            tokenized_dataset.save_to_disk(output_path)
            print("{}に保存完了".format(output_path))
            
            """
            if (((self.dataset_nums - 2) < data_set_num) and (data_set_num < self.dataset_nums)) :
                self.last_dataset_flag = True
            """

            if self.last_dataset_flag == True :
                break


    def tokenize_files(
        self, data_set_num, data_directory, file_format: Literal["loom", "h5ad"] = "loom"
    ):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            # loomfileのcolumns名(細胞名)をリストに格納
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            # トークン化するcolumns名(細胞名)をkeyとして[]をvalueとした辞書型を生成（初期化）
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        # loops through directories to tokenize .loom files
        # loomfileが存在するかのフラグ 0は存在しない．1は存在する．
        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        # loomfileのtokenizer関数を選択し，格納する
        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        )

        # loomfile全部をfor-loopで参照
        total_loom_datas = len(glob.glob(data_directory+"*.loom"))
        start_time1 = time()
        total_cells_num = 0
        loaded_loomfile_num = 0
        for enum1, file_path in enumerate(Path(data_directory).glob("*.{}".format(file_format))): # file_format = loom format
        # for enum1, file_path in enumerate()
            if (enum1 < self.start_reading_file_num) : 
                continue
            # loomfileが存在するかのフラグ fileは存在する．
            file_found = 1
            start_time2 = time()
            print("=================================")
            print("[{} / {}]".format(enum1, total_loom_datas))
            print("Tokenizing : {}".format(file_path))
            
            # tokenize_loom関数を実行
            # ==== 戻り値 ====
            # file_tokenized_cells：loomfile内の細胞の正規化発現量が大きい順になったEnsembl IDのトークンIDが格納されている．
            # file_cell_metadata：loomfileの細胞に関するメタデータ．self.custom_attr_name_dict = Noneなため，file_cell_metadata = Noneとなる．
            file_tokenized_cells, file_cell_metadata, cells_num, not_use_file_name = tokenize_file_fn(file_path) #tokenize_loom()
            # 全loomfileの細胞のトークンIDを格納
            tokenized_cells += file_tokenized_cells
            #print("====")
            #print(tokenized_cells)
            #print(type(tokenized_cells))
            #print("====")

            """
            if not_use_file_name is not None :
                with open("/mnt/keita/data/scRNA-data/normal_mouse/mouse_datas/tmp/not_use_tokenized_cells_file.csv", mode="a") as f:
                    f.write(not_use_file_name+"\n")
            """


            # loomfileの細胞に対してのmetadataを格納．self.custom_attr_name_dict = Noneなため，cell_metadata = None
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
            else:
                cell_metadata = None

            loaded_loomfile_num += 1
            total_cells_num += cells_num
            
            #print("1 dataset in {} loom files".format(int((total_loom_datas - self.end_file_nums) / (self.dataset_nums - 17))))
            print("current total cells : {}".format(total_cells_num))
            print("Number of loaded loom files : {} ".format(loaded_loomfile_num))
            print(f"{file_path} Tokenizing time[s]: {time()-start_time2}")
            print(f"Time token to create 1 dataset[s]: {time()-start_time1}")
            print(f"Time toking to create {self.dataset_nums} dataset[s]: {time()-self.starting_program_time}")


            """
            if (1 < self.dataset_nums) and (self.last_dataset_flag == False):
                if ((((int((total_loom_datas - self.end_file_nums) / (self.dataset_nums - 0)) * (data_set_num - 0)) + self.end_file_nums) - 2) < enum1) :
                #if (((int(total_loom_datas / self.dataset_nums) * data_set_num) - 2) < enum1) :
                    print("{}のうち1つのデータセットを作成".format(self.dataset_nums))
                    self.start_reading_file_num = enum1 + 1
                    break
                else :
                    pass
            else :
                pass
            """

            if enum1 == total_loom_datas-1 :
                print("{}番目のデータセットは最後のデータセット".format(enum1))
                self.last_dataset_flag = True
                break
            else :
                if total_cells_num > self.max_cells_of_one_dataset :
                    print("{}から{}まで（計{}個）のファイルでデータセットを作成".format(self.start_reading_file_num, enum1, loaded_loomfile_num))
                    self.start_reading_file_num = enum1 + 1
                    print("次に読み込むファイルは{}番目のファイル".format(self.start_reading_file_num))
                    break
                else :
                    pass
            
            
            

            """
            # テスト
            if enum1 > 4 :
                break
            """



        # fileが存在しなかった場合の処理
        if file_found == 0:
            logger.error(
                f"No .{file_format} files found in directory {data_directory}.")
            raise
        
        # tokenized_cells：全loomfileの全細胞のトークン
        # loomfileの細胞に対してのmetadataを格納．self.custom_attr_name_dict = Noneなため，cell_metadata = None
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
            # loomfileのcolumns名(細胞名)をkeyとして[]をvalueとした辞書型を生成（初期化）
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys() 
            }
        
        file_name = None

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            # loomfileのensembl_idのindexをとってきている
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["row_attrs"]] # "ensembl_id"を"row_attrs"に変更する必要がある
            )[0]
            # loomfileのensembl_idの中央値をとってきている
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["row_attrs"][coding_miRNA_loc]
                ]
            )
            # loomfileのensembl_idを格納している
            coding_miRNA_ids = data.ra["row_attrs"][coding_miRNA_loc]
            # loomfileのensembl_idをkeysとしたときのvalueであるtoken_idをとってきている
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )
            print(coding_miRNA_tokens.shape[0])
                
            #print([type(coding_miRNA_tokens[i]) for i in range(coding_miRNA_tokens.shape[0])])

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            # data.ca["filter_pass"]は存在しないため，exceptが実行され，var_exists = Falseとなる
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
                # loomfileの全ての細胞がトークンか対象となる
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                cells_counts = data.shape[1]
                filter_pass_loc = np.array([i for i in range(cells_counts)])

            # scan through .loom files and tokenize cells
            # loomfileをスキャンし，loomfile内の細胞をトークン化する
            tokenized_cells = []
            # itemsに格納された条件に当う列（細胞）を全て取得．故に，for-loopは1回である．
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                # フィルタリング後の細胞のensembl_idがある遺伝子をとってきている
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                # ランク値エンコーディング：正規化発現量の計算
                # 式：(subview[:,:] * (1 / np.sum(subview, axis=1)) * 10000 * (1 / norm_factor_vector[:, None]))
                subview_norm_array = (
                    subview[:, :]                  # loomfile内の細胞の遺伝子発現量
                    / np.sum(subview[:, :], axis=0)#subview.ca.n_counts          # loomfile内の細胞の遺伝子発現総数で正規化
                    * target_sum                   # 10000倍
                    / norm_factor_vector[:, None]  # loomfileにあるEnsembl ID（遺伝子）の中央値で正規化
                )
                # tokenize subview gene vectors
                # ランク値エンコーディング：ランク値に基づいて大きい順にした遺伝子をトークン化
                if coding_miRNA_tokens.shape[0] > 0 :
                    tokenized_cells += [
                        tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens) # tokenize_cell(loomfileの1細胞の正規化発現量, loomfileのensembl_idのtoken_id)
                        for i in range(subview_norm_array.shape[1])
                    ]
                else :
                    file_name = str(loom_file_path)

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    # loomfileのcolumns名(細胞名)のvalueにフィルタリング後の細胞のensembl_idがある遺伝子をもってきてる(トークン化する細胞を持ってきている)
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                else:
                    file_cell_metadata = None
        
        # tokenized_cells：loomfile内の細胞の正規化発現量が大きい順になったEnsembl IDのトークンIDが格納されている．
        # file_cell_metadata：loomfileの細胞に関するメタデータ．self.custom_attr_name_dict = Noneなため，file_cell_metadata = Noneとなる．

        return tokenized_cells, file_cell_metadata, cells_counts, file_name

    def create_dataset(self, tokenized_cells, cell_metadata, use_generator=False):
        print("Creating dataset.")
        # create dict for dataset creation
        # トークンを辞書型にして格納
        # dataset_dict：keyに"input_ids"，valueに全loomfileの全細胞のトークン(list)を持つ辞書型
        
        """
        print("====")
        print(len(tokenized_cells))
        print(type(tokenized_cells))
        print("====")
        """


        # 外部ファイルに保存してあるtokenized_cellsに格納している全loomfileの細胞のトークンIDを取得(list型)
        # tokenized_cellsを取得するときは，tokenized_cellsのデータ構造のまま取得    
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None: # skip
            dataset_dict.update(cell_metadata)

        # create dataset
        if use_generator: # use_generator = Falseなため，skip
            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}
            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            # appach arrow形式の.datasetにdataset_dictを保存
            #output_dataset = Dataset.from_dict(dataset_dict)
            output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset
        # 各細胞のトークンが2048以下のトークン数になるように整形
        
        def truncate(example):
            example["input_ids"] = example["input_ids"][:2048]
            return example
        
        
        # datasetの各細胞のトークンが2048以下になったものを追加 or 保存
        output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)
        

        # measure lengths of dataset
        # dataset全体の大きさを計算
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example
        
        # dataset全体の大きさを計算し，datasetに追加 or 保存
        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=self.nproc
        )

        # datasetの各細胞のトークンが2048以下となり，dataset全体の大きさを追加したdatasetをreturn
        return output_dataset_truncated_w_length



class Cell_Type_Classification_TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict=None,
        nproc=1, # 4にしたほうがいい?
        dataset_num=-1,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : None, dict
            Dictionary of custom attributes to be added to the dataset.    データセットに追加するカスタム属性の辞書。
            Keys are the names of the attributes in the loom file.         キーはloomファイルの属性名。
            Values are the names of the attributes in the dataset.         値はデータセットの属性名。
        nproc : int
            Number of processes to use for dataset mapping.
        gene_median_file : Path
            Path to pickle file containing dictionary of non-zero median
            gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl IDs:token).
        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        # keyがloomfileのcolumns名(細胞名)でvalueがトークン化するcolumns名(細胞名)となる辞書(その辞書があれば辞書を格納するが，なければNoneを格納)
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        # 非ゼロ中央値ファイルの中身を保存
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        # トークンファイルの中身を保存
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary
        # 非ゼロ中央値のEnsembl ID数+2(特殊文字2個)を格納
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        # loomfileの細胞をトークン化するために，特定のEnsembl ID行を取得するための辞書作成
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

        # トークン化するために読み始めるloomfileのインデックス番号
        # defaultの値は-1
        # self.start_reading_file_num = -1    : はじめのファイルから読む
        # self.start_reading_file_num = (1~x) : (1~x)のファイルから読む
        self.start_reading_file_num = -1

        # 読み終えたファイルの個数
        self.end_file_nums = 0

        # appach arrow形式のトークンのデータセット数(1つのdatasetfileにメモリの関係でトークンデータを保存できないときに使用)
        # 1つのdatasetfileに保存する細胞数は「学習させる総細胞数 / self.dataset_nums」個の細胞となる
        # defaultの値は1
        self.dataset_nums = dataset_num

        # 学習させる総細胞数
        self.learning_cell_nums = 22_446_161

        # 最後に作成するデータセットかどうかを判断するフラグ
        self.last_dataset_flag = False

        # 1datasetに含む細胞数の最大値
        self.max_cells_of_one_dataset = 300_000

        self.starting_program_time = time()

        print("=================================")
        print("Setting")
        print("TranscriptomeTokenizer")
        print("custom_attr_name_dict = {}".format(self.custom_attr_name_dict))
        print("gene_median_file = {}".format(gene_median_file))
        print("token_dictionary_file = {}".format(token_dictionary_file))
        print("nproc = {}".format(self.nproc))
        print("dataset_nums = {}".format(self.dataset_nums))
        print("learning_cell_nums = {}".format(self.learning_cell_nums))
        print("planned_dataset_nums = {}".format(int(self.learning_cell_nums / self.dataset_nums)+1))

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
            Path to directory containing loom files or anndata files                loom ファイルまたは anndata ファイルを含むディレクトリへのパス。
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset        .dataset形式のファイルを保存するパス
        output_prefix : str
            Prefix for output .dataset                                               出力する.datasetの接頭辞
        file_format : str
            Format of input files. Can be "loom" or "h5ad".                          入力ファイルのフォーマット。loom "または "h5ad"。
        use_generator : bool
            Whether to use generator or dict for tokenization.                       トークン化にジェネレーターを使うかdictを使うか
        """

        for data_set_num in range(0+1,84+1,1) :

            if data_set_num < 0 :
                continue
            
            # ランク値エンコーディングを実行し，エンコーディング結果と細胞のメタデータを格納
            # ==== 戻り値 ====
            # tokenized_cells：全loomfileの全細胞のトークン(list型[入れ子])
            # cell_metadata：loomfileの細胞に関するメタデータ．self.custom_attr_name_dict = Noneなため，file_cell_metadata = Noneとなる．
            tokenized_cells, cell_metadata = self.tokenize_files(
                data_set_num, data_directory, file_format
            )
            if int(len(tokenized_cells)) == 0 :
                continue
            
            # エンコーディング結果と細胞のメタデータからarrow形式のトークン化したデータセットを作成
            # == 戻り値 ==
            # datasetの各細胞のトークン数が2048以下となり，dataset全体の大きさを追加したappach arrow形式のdataset
            tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata, use_generator=use_generator)
            
            # 作成したappach arrow形式のトークン化したデータセットを保存するパスを作成
            #output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
            output_path = output_directory+"/"+output_prefix+"_"+str(data_set_num)+".dataset"
            # パスに作成したarrow形式のトークン化したデータセットを保存
            tokenized_dataset.save_to_disk(output_path)
            print("{}に保存完了".format(output_path))
            
            """
            if (((self.dataset_nums - 2) < data_set_num) and (data_set_num < self.dataset_nums)) :
                self.last_dataset_flag = True
            """

            if self.last_dataset_flag == True :
                break


    def tokenize_files(
        self, data_set_num, data_directory, file_format: Literal["loom", "h5ad"] = "loom"
    ):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            # loomfileのcolumns名(細胞名以外)をリストに格納
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            # トークン化するcolumns名(細胞名以外)をkeyとして[]をvalueとした辞書型を生成（初期化）
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        # loops through directories to tokenize .loom files
        # loomfileが存在するかのフラグ 0は存在しない．1は存在する．
        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        # loomfileのtokenizer関数を選択し，格納する
        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        )

        # loomfile全部をfor-loopで参照
        total_loom_datas = len(glob.glob(data_directory+"*.loom"))
        start_time1 = time()
        total_cells_num = 0
        loaded_loomfile_num = 0
        for enum1, file_path in enumerate(Path(data_directory).glob("*.{}".format(file_format))): # file_format = loom format
        # for enum1, file_path in enumerate()
            if (enum1 < self.start_reading_file_num) : 
                continue
            # loomfileが存在するかのフラグ fileは存在する．
            file_found = 1
            start_time2 = time()
            print("=================================")
            print("[{} / {}]".format(enum1, total_loom_datas))
            print("Tokenizing : {}".format(file_path))
            
            # tokenize_loom関数を実行
            # ==== 戻り値 ====
            # file_tokenized_cells：loomfile内の細胞の正規化発現量が大きい順になったEnsembl IDのトークンIDが格納されている．
            # file_cell_metadata：loomfileの細胞に関するメタデータ．self.custom_attr_name_dict = Noneなため，file_cell_metadata = Noneとなる．
            file_tokenized_cells, file_cell_metadata, cells_num, not_use_file_name = tokenize_file_fn(file_path) #tokenize_loom()
            # 全loomfileの細胞のトークンIDを格納
            tokenized_cells += file_tokenized_cells
            #print("====")
            #print(tokenized_cells)
            #print(type(tokenized_cells))
            #print("====")

            """
            if not_use_file_name is not None :
                with open("/mnt/keita/data/scRNA-data/normal_mouse/mouse_datas/tmp/not_use_tokenized_cells_file.csv", mode="a") as f:
                    f.write(not_use_file_name+"\n")
            """


            # loomfileの細胞に対してのmetadataを格納．self.custom_attr_name_dict = Noneなため，cell_metadata = None
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    # cell_attr = ["cell_types", "organ_major"]
                    # self.custom_attr_name_dict = {"cell_types":"cell_type", "organ_major":"organ_major"}
                    # cell_metadata = {"cell_type":[], "organ_major":[]}
                    # file_cell_metadata = {"cell_types":[cell_type1, cell_type2, ...., cell_typeN], "organ_major":["kidney", "kidney", "kidney", "kidney", ..."kidney"]}
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
                    # cell_metadata = {"cell_type": [cell_type1, cell_type2, ...., cell_typeN]}
            else:
                cell_metadata = None

            loaded_loomfile_num += 1
            total_cells_num += cells_num
            
            #print("1 dataset in {} loom files".format(int((total_loom_datas - self.end_file_nums) / (self.dataset_nums - 17))))
            print("current total cells : {}".format(total_cells_num))
            print("Number of loaded loom files : {} ".format(loaded_loomfile_num))
            print(f"{file_path} Tokenizing time[s]: {time()-start_time2}")
            print(f"Time token to create 1 dataset[s]: {time()-start_time1}")
            print(f"Time toking to create {self.dataset_nums} dataset[s]: {time()-self.starting_program_time}")


            """
            if (1 < self.dataset_nums) and (self.last_dataset_flag == False):
                if ((((int((total_loom_datas - self.end_file_nums) / (self.dataset_nums - 0)) * (data_set_num - 0)) + self.end_file_nums) - 2) < enum1) :
                #if (((int(total_loom_datas / self.dataset_nums) * data_set_num) - 2) < enum1) :
                    print("{}のうち1つのデータセットを作成".format(self.dataset_nums))
                    self.start_reading_file_num = enum1 + 1
                    break
                else :
                    pass
            else :
                pass
            """

            if enum1 == total_loom_datas-1 :
                print("{}番目のデータセットは最後のデータセット".format(enum1))
                self.last_dataset_flag = True
                break
            else :
                if total_cells_num > self.max_cells_of_one_dataset :
                    print("{}から{}まで（計{}個）のファイルでデータセットを作成".format(self.start_reading_file_num, enum1, loaded_loomfile_num))
                    self.start_reading_file_num = enum1 + 1
                    print("次に読み込むファイルは{}番目のファイル".format(self.start_reading_file_num))
                    break
                else :
                    pass
            
            
            

            """
            # テスト
            if enum1 > 4 :
                break
            """



        # fileが存在しなかった場合の処理
        if file_found == 0:
            logger.error(
                f"No .{file_format} files found in directory {data_directory}.")
            raise
        
        # tokenized_cells：全loomfileの全細胞のトークン
        # loomfileの細胞に対してのmetadataを格納．self.custom_attr_name_dict = Noneなため，cell_metadata = None
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
            # loomfileのcolumns名(細胞名)をkeyとして[]をvalueとした辞書型を生成（初期化）
            file_cell_metadata = {
                # self.custom_attr_name_dict = {"cell_types":"cell_type", "organ_major":"organ_major"}
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys() 
            }
            # file_cell_metadata = {"cell_types": [], "organ_major": []}
        file_name = None

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            # loomfileのensembl_idのindexをとってきている
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["row_attrs"]] # "ensembl_id"を"row_attrs"に変更する必要がある
            )[0]
            # loomfileのensembl_idの中央値をとってきている
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["row_attrs"][coding_miRNA_loc]
                ]
            )
            # loomfileのensembl_idを格納している
            coding_miRNA_ids = data.ra["row_attrs"][coding_miRNA_loc]
            # loomfileのensembl_idをkeysとしたときのvalueであるtoken_idをとってきている
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )
            print(coding_miRNA_tokens.shape[0])
                
            #print([type(coding_miRNA_tokens[i]) for i in range(coding_miRNA_tokens.shape[0])])

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            # data.ca["filter_pass"]は存在しないため，exceptが実行され，var_exists = Falseとなる
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
                # loomfileの全ての細胞がトークンか対象となる
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                cells_counts = data.shape[1]
                filter_pass_loc = np.array([i for i in range(cells_counts)])

            # scan through .loom files and tokenize cells
            # loomfileをスキャンし，loomfile内の細胞をトークン化する
            tokenized_cells = []
            # itemsに格納された条件に当う列（細胞）を全て取得．故に，for-loopは1回である．
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                # フィルタリング後の細胞のensembl_idがある遺伝子をとってきている
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                # ランク値エンコーディング：正規化発現量の計算
                # 式：(subview[:,:] * (1 / np.sum(subview, axis=1)) * 10000 * (1 / norm_factor_vector[:, None]))
                subview_norm_array = (
                    subview[:, :]                  # loomfile内の細胞の遺伝子発現量
                    / np.sum(subview[:, :], axis=0)#subview.ca.n_counts          # loomfile内の細胞の遺伝子発現総数で正規化
                    * target_sum                   # 10000倍
                    / norm_factor_vector[:, None]  # loomfileにあるEnsembl ID（遺伝子）の中央値で正規化
                )
                # tokenize subview gene vectors
                # ランク値エンコーディング：ランク値に基づいて大きい順にした遺伝子をトークン化
                if coding_miRNA_tokens.shape[0] > 0 :
                    tokenized_cells += [
                        tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens) # tokenize_cell(loomfileの1細胞の正規化発現量, loomfileのensembl_idのtoken_id)
                        for i in range(subview_norm_array.shape[1])
                    ]
                else :
                    file_name = str(loom_file_path)

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    # loomfileのcolumns名(細胞名)のvalueにフィルタリング後の細胞のensembl_idがある遺伝子をもってきてる(トークン化する細胞を持ってきている)
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                        # file_cell_metadata["cell_types"] = [cell_type1, cell_type2, cell_type3, ..., cell_typeN]
                        # file_cell_metadata["organ_major"] = ["kidney", "kidney", "kidney", "kidney", ...,"kidney"]
                else:
                    file_cell_metadata = None
        
        # tokenized_cells：loomfile内の細胞の正規化発現量が大きい順になったEnsembl IDのトークンIDが格納されている．
        # file_cell_metadata：loomfileの細胞に関するメタデータ．self.custom_attr_name_dict = Noneなため，file_cell_metadata = Noneとなる．

        return tokenized_cells, file_cell_metadata, cells_counts, file_name

    def create_dataset(self, tokenized_cells, cell_metadata, use_generator=False):
        print("Creating dataset.")
        # create dict for dataset creation
        # トークンを辞書型にして格納
        # dataset_dict：keyに"input_ids"，valueに全loomfileの全細胞のトークン(list)を持つ辞書型
        
        """
        print("====")
        print(len(tokenized_cells))
        print(type(tokenized_cells))
        print("====")
        """


        # 外部ファイルに保存してあるtokenized_cellsに格納している全loomfileの細胞のトークンIDを取得(list型)
        # tokenized_cellsを取得するときは，tokenized_cellsのデータ構造のまま取得    
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None: # skip
            dataset_dict.update(cell_metadata)
            # dataset_dict = {"input_ids": tokened_cells, "cell_type": [cell_type1, cell_type2, ...., cell_typeN], "organ_major": ["kidney", "kidney", ..."kidney"]}

        # create dataset
        if use_generator: # use_generator = Falseなため，skip
            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}
            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            # appach arrow形式の.datasetにdataset_dictを保存
            #output_dataset = Dataset.from_dict(dataset_dict)
            output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset
        # 各細胞のトークンが2048以下のトークン数になるように整形
        
        def truncate(example):
            example["input_ids"] = example["input_ids"][:2048]
            return example
        
        
        # datasetの各細胞のトークンが2048以下になったものを追加 or 保存
        output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)
        

        # measure lengths of dataset
        # dataset全体の大きさを計算
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example

        # dataset全体の大きさを計算し，datasetに追加 or 保存
        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=self.nproc
        )
        
        # 細胞タイプを格納
        # def cell_type(example) :
        #    example["cell_type"] = 細胞ごとにアノテーションされた細胞タイプ
        #    return example  

        #output_dataset_truncated_w_length_w_type = output_dataset_truncated_w_length.map(
        #    cell_type, num_proc=self.nproc
        #)
        
        # datasetの各細胞のトークンが2048以下となり，dataset全体の大きさを追加したdatasetをreturn
        return output_dataset_truncated_w_length
