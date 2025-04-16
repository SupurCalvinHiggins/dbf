# dbf
🔬 Experiments with Downtown Bodega filters

# Background

Downtown Bodega filters (DBFs) [1] are a secure variant of Learned Bloom filters (LBFs) [2]. DBFs use Naor-Eylon filters [3], a secure Bloom filter variant, to verify the output of a learned model. See `writing/proposal.pdf` for some more details.

# Overview

This repository includes implementations of
1. The Downtown Bodega filter data structure
2. An optimal Downtown Bodega filter parameter tuning algorithm
3. Three adaptive attacks against the Downtown Bodega filter

for my machine learning security final project (CSC 592).

# Installation

Installation is easy(!) because of the **amazing** `uv` package manager.

1. Install the [uv package manager](https://github.com/astral-sh/uv).
2. Install the required libraries by running `uv sync` in the repository root.

## System Requirements

The code was tested on Linux with a V100 GPU. Performance might be poor if the GPU does not have Tensor Cores.

# Quick Start

View the results by running 
```sh
uv run attack.py
```
Expect to wait around five minutes. If you need more speed, vectorize the parameter tuning algorithm and look into a GPU implementation of BLAKE3.

Optionally re-train the models by running
```sh
rm -rf model/*.pt
uv run train.py
uv run attack.py
```

# Implementation

The implementation is a __mess__. It got hacked together over the course of a weekend. If something doesn't make sense to you, it probably doesn't make sense at all. 

## Naor-Eylon Filter

Instead of the main Naor-Eylon filter construction discussed in [1] and [3] based on pseudo-random permutations, I opt to use the construction based on pseudo-random functions mentioned in [3]. Basically, to create the Naor-Eylon filter, the hash function of a standard Bloom filter is replaced with a cryptographic hash function. I use the `BLAKE3` hash function.

## Downtown Bodega Filter

The Downtown Bodega filter automatically selects optimal (lowest memory footprint) false positive rates for the backup filters, given a fixed threshold.

### Parameter Tuning

The parameter tuning algorithm is explained in detail in `writing/tuning_guide.pdf`. In short, the algorithm finds the optimal parameters to minimize the DBF's memory footprint subject to expected and worst-case bounds on the DBF's false positive rate. The algorithm is easily __vectorizable__ but I haven't found the time to work through the full implementation. 

# Experiments

## Dataset

I store URLs from [a dataset of malicious URLs](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) in the DBF. URLs are truncated to 128 bytes. Truncation only impacts a few URLs. Run `uv run stats.py` to see for yourself.

## Downtown Bodega Filter

I train a small single layer Transformer on a binary URL classification task (malicious/benign). I build a Downtown Bodega Filter with expected false positive rate 0.05 and maximum false positive rate 0.20 with optimal parameters. I also build a Naor-Eylon filter with the same memory budget.

## Adaptive Attacks

I attack the Downtown Bodega Filter with three different adaptive attacks. The attacks aim to generate unseen examples that induce a high false positive rate.

### Randomized Attack

I generate random unseen URLs and pass them through the DBF. 

### Adaptive Black-Box Attack

I train a synthetic Transformer to predict if a URL induces a false positive in the DBF. Then, I optimize the URLs from the randomized attack using Adam for the false positive label. I discard previously seen URLs, and pass the adversial URLs through the DBF. I also run a McNemar test to compare this attack against the randomized attack. 

### Targeted Randomized Attack

I split the URLs from the randomized attack into two sets based on the learned model prediction. I pass the set corresponding to the "weaker" filter to the DBF.

# References

[1] "Adversary Resilient Learned Bloom Filters" by Allison Bishop and Hayder Tirmazi.

[2] "The Case for Learned Index Structures” by Tim Kraska et al.

[3] "Bloom Filters in Adversarial Environments" by Moni Naor and Yogev Eylon.
