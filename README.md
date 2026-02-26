# AttentionDCA

Package used for the analysis of the factored self-attention mechanism through a simple one-layer DCA model at:

1. Caredda F., Gennai L., De Los Rios P., Pagnani A., Controllable protein design via autoregressive direct coupling analysis conditioned on principal components, [Plos Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013996)

## Install

This is an unregistered package: to install enter `]` in the Julia repl and

```
pkg> add https://github.com/caredda/FeatureDCA.jl

```
## Use

The functions for the training and sampling are
```
trainer, sample
```
These take as as inputs either tuples with integer-encoded MSA, weight vector, and feature matrix $(Z,W,Y)$ or a path to the fasta file containing the sequences of the protein family under study along with the feature vector Y. To get more details on the use of e$
```
?trainer
```

## Data

All data used in this study is publicly available at [GitHub/francescocaredda/FeatureDCAData](https://github.com/francescocaredda/FeatureDCAData) in the "data" folder.


## Inquiries

Any question can be directed to francesco.caredda@polito.it
