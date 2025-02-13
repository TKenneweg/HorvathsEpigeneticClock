# Implementing Horvath's Epigenetic Clock From Scratch

This repository contains the code for the YouTube video **"Implementing Horvath's Epigenetic Clock From Scratch."** ([Link](https://youtu.be/ZS0_b2KWQos)).

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

before running download.py followed by preprocess.py to get the actual data.

You need to run mlp.py before regression.py since regression.py tries to read train/test split indices from disk.