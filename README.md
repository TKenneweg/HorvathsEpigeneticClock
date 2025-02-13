# WhyDoWeAge

This repository contains the code for the YouTube video **"Why Do We Age? Exploring the Evolutionary Causes of Aging."** ([Link](https://youtu.be/cjHC1akKCVI)).

## Getting Started

Start with the usual steps to run the code:

```bash
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

Afterwards make sure to create the data folder 
```bash
mkdir data
```

Before running download.py followed by preprocess.py.

Before running regression.py make sure you run mlp.py so that some train-test split indices get generated.