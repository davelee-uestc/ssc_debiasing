# Mitigating Unknown Bias in Model by Suppressing Spurious Features

This repository contains experiments for the paper _Mitigating Unknown Bias in Model by Suppressing Spurious Features_
## Introduction
Deep neural networks (DNNs) often learn spurious correlations, also known as shortcut learning, leading to incorrect predictions in critical applications such as automated diagnosis. Traditional methods require prior knowledge of the bias attributes or retraining models, limiting their applicability. This paper presents a novel debiasing method, SSC, which assumes that biases arise from unintended patterns in the training data due to sampling bias. SSC applies a learned linear transformation after the encoder to neutral the bias attributes and lower the coefficient of bias attributes in the classifier to remove the impact of spurious features without altering the encoder. Experimental results across multiple benchmarks demonstrate that SSC significantly improves performance on tasks affected by spurious features, achieving state-of-the-art results in both vision and text tasks. Ablation studies and qualitative analyses further validate the proposed method.


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
