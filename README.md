# Velocount

Count spliced and unspliced reads from single-cell RNA-seq for downstream RNA velocity analysis.
This was forked from the original [velocyto.py](https://github.com/velocyto-team/velocyto.py) repository, simplified
to remove any code related to analysis, and updated to use versioned Python libraries.

This probably *should* still work on data from Smart-seq2 or general scRNA-seq data, but I have **only** tested this with 
data from 10X Genomics assays.

At current, install using:
```
pip install git+https://github.com/milescsmith/velocyto.py@simplify
```

Run
```
velocount run --help
```
for more information on how to run.