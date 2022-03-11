# Align-and-Predict
Codes for the paper "Adjusting the Precision-Recall Trade-Off with Align-and-Predict Decoding for Grammatical Error Correction" (ACL 2022)


## Installation

```
conda create -n ADP python=3.6
conda activate ADP
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
pip install fairseq
```

## Usage
This section explains how to decode in different ways.
```
PTPATH=/to/path/checkpoint*.pt # path to model file
BINDIR=/to/path/bin_data # directory containing src and tgt dictionaries  
INPPATH=/to/path/bea*.bpe.txt # path to eval file
OUTPATH=/to/path/bea*.out.txt # path to output file
BATCH=xxx
BEAM=xxx
RATIO=xxx
```


## Beam search (baseline):

```
python inference.py --checkpoint-path $PTPATH --bin-data $BINDIR --input-path $INPPATH --output-path $OUTPATH --batch $BATCH --beam $BEAM --baseline
```

## Align-and-Predict:

```
python inference.py --checkpoint-path $PTPATH --bin-data $BINDIR --input-path $INPPATH --output-path $OUTPATH --batch $BATCH --beam $BEAM --ratio $RATIO
```
