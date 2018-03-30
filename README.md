# R-STEINER

This repository provides the scripts for [R-STEINER](https://ipsj.ixsq.nii.ac.jp/ej/index.php?active_action=repository_view_main_item_detail&page_id=13&block_id=8&item_id=186582&item_no=1).
The R-STEINER is discovery method of translation enhancers, i.e. we can obtain 5'UTRs which increase the translated proteins &mdash or, translation efficiency &mdash by R-STEINER.
The setting of the scripts is based on [Tanaka et al. (2018)](https://ipsj.ixsq.nii.ac.jp/ej/index.php?active_action=repository_view_main_item_detail&page_id=13&block_id=8&item_id=186582&item_no=1), i.e. CDS and 3'UTR are given.


## Directory

We have to adopt following directories.

```
R-STEINER/
  Scripts/  # the scripts of R-STEINER is in this directory
    mk-predictors.py
    rsteiner.py
    settenQBmodule.py
  Data/  # the original datasets should be preserved in this directory
    sequence data in HS condition (pandas' pickle file)
    sequence data in Con condition (pandas' pickle file)
    PR-value data in HS condition (pandas' pickle file)
    PR-value data in Con condition (pandas' pickle file)
    feature/  # the features used in R-STEINER is preserved
  Models/  # the models used in R-STEINER is preserved
```


## How to Use

1. You have to make datasets which are used to build the prediction model of PR-value; both of Heat Stress condition and Controlled condition are required &mdash detail is described in the paper.
All of the raw CAGE sequences data used in our paper are available on the [DDBJ Sequence Read Archive (DRA) database with accession number DRA006661](https://trace.ddbj.nig.ac.jp/DRASearch/submission?acc=DRA006661).
You can transform these data to the sequence data which can be used actually by procedure shown in Section 4.1.3 of the paper.
You have to preserve the transformed data into `R-STEINER/Data`.
The form of datasets of mRNA sequences is the following.

| Gene ID | 5'UTR |  CDS  | 3'UTR |
|:--------|:------|:------|:------|
| 1       |TTC... |ACC... | CCG...|


2. You have to install a software: [ViennaRNA Package](https://www.tbi.univie.ac.at/RNA/).

3. You should run the following programs:
    1. `mk-predictors.py`
    1. `rsteiner.py`

Then you can obtain top-$k$ translation enhancers in `R-STEINER/Data/result_R-STEINER.csv`.
