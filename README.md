# Mitigating Unknown Bias by Suppressing Spurious Features

This repository contains experiments for the paper _Mitigating Unknown Bias by Suppressing Spurious Features_
## Introduction
Removing unintended spurious correlations learned in Deep neural networks (DNNs), also known as shortcut learning, is crucial for critical applications such as automated diagnosis.  In this study, we introduce a novel debiasing method Suppressing Spurious Features (SSF). Unlike previous methods, SSF assumes that biases arise from unintended patterns in the training data due to sampling bias, enabling estimating the unbiased mean of spurious features to neutralize them without prior knowledge of bias attributes. Specifically, SSF learns a linear transformation after the encoder to neutralize the bias attributes, then lowers their coefficients in the classifier, thus the impact of spurious features is removed without retraining the encoder. Experimental results across multiple vision and text benchmarks demonstrate that SSF significantly improves the performance of models affected by spurious features, achieving state-of-the-art. Ablation studies and qualitative analyses validate the effectiveness of SSF in removing spurious features and reducing bias.

## File Structure

```
.
+-- train_supervised.py (Train base models)
+-- ssc.py (Main Experiments)
+-- ssc_common.py (Common utilities used in different scripts)
```

## Requirements

- [`torch`](https://pytorch.org/get-started/locally/)
- [`torchvision`](https://pytorch.org/get-started/locally/)
- [`timm`](https://github.com/rwightman/pytorch-image-models)
- [`transformers`](https://huggingface.co/docs/transformers/installation)
- [`vissl`](https://github.com/facebookresearch/vissl/blob/main/INSTALL.md)
- [`scikit-learn`](https://scikit-learn.org/stable/install.html)
- [`numpy`](https://numpy.org/install/)
- [`tqdm`](https://pypi.org/project/tqdm/)
- [`wilds`](https://wilds.stanford.edu/get_started/)


## Data access

### Waterbirds and CelebA

Please follow the instructions in the [DFR repo](https://github.com/PolinaKirichenko/deep_feature_reweighting#data-access) to prepare the Waterbirds and CelebA datasets in the `./waterbirds`.

### Civil Comments and MultiNLI

The Civil Comments dataset should be downloaded automatically when you run experiments, no manual preparation needed.

To run experiments on the MultiNLI dataset, please manually download and unzip the dataset from [this link](https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz) in the `./multinli`.
Further, please copy the `dataset_files/utils_glue.py` to the root directory of the dataset.

### CheXpert

The CheXpert dataset are not publically available, so we share the code `medical_dataset_preprocess.py` for preparing this dataset.


## Example commands

Waterbirds:
```bash
./train_sup.sh -d wb
python3 ssc.py waterbirds 
```

CelebA:
```bash
./train_sup.sh -d ce
python3 ssc.py celeba 
```


Civil Comments
```bash
./train_sup.sh -d cc
python3 ssc.py multinli 
```

MultiNLI
```bash
./train_sup.sh -d mn
python3 ssc.py civilcomments 

```

CheXpert
```bash
./train_sup.sh -d cx
python3 ssc.py chexpert 

```


## Code References

- We used the [DFR codebase](https://github.com/izmailovpavel/spurious_feature_learning) as the basis for our code.

- Our model implementations are based on the `torchvision`, `timm` and `transformers` packages.
