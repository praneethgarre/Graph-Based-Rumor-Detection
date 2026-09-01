# Graph-Based Rumor Detection

A graph neural network approach for rumor detection on the PHEME dataset.

## Overview

The system combines:

- **BERT (`bert-base-cased`)** for tweet text representations
- **User metadata** such as verification status, followers, friends, tweet count, and account age
- **Graph Attention Network (GAT)** to model relationships between tweets in a conversation
- **Global mean pooling** for graph-level representation
- A fully connected neural network for binary rumor/non-rumor classification

## Architecture

```text
Tweet Text
    |
    v
BERT (768)
    |
    +-------------------+
    |                   |
User Features           |
(5 features)            |
    |                   |
    +------ Concatenate-+
             |
             v
      Linear Projection
          773 -> 712
             |
             v
        GAT Layer
        712 -> 1024
             |
             v
     Global Mean Pooling
             |
             v
       Fully Connected
   1024 -> 987 -> 463
        -> 654 -> 1
             |
             v
          Sigmoid
             |
             v
     Rumor / Non-Rumor
```

## Dataset

The implementation uses the PHEME rumor-detection dataset.

The script downloads the dataset using KaggleHub and constructs one graph
for each conversation thread. The original implementation processed 9 event
folders and created 6,425 graphs.

## Node Features

Each graph node represents a tweet and contains:

1. BERT `[CLS]` text embedding — 768 dimensions
2. Verified status — 1 dimension
3. Followers count — 1 dimension
4. Friends count — 1 dimension
5. Tweet/status count — 1 dimension
6. Account age in days — 1 dimension

Total: **773 node features**

These features are projected to **712 dimensions** before the GAT layer.

## Graph Construction

The source tweet is used as the root node.

Replies/reactions are added as additional nodes. Reply relationships are
represented using bidirectional edges.

Node features are normalized using Min-Max scaling.

## Training

- Optimizer: Adam
- Learning rate: `0.0000313`
- Weight decay: `0.00221`
- Loss: Binary Cross Entropy
- Batch size: 32
- Epochs: 30
- Random seed: 42

The data is split using stratified sampling:

- 90% Train + Validation
- 10% Test
- 80% Train
- 20% Validation

The training/validation pool is balanced by creating masked copies of
minority rumor graphs.

## Results

The provided experiment reported:

| Metric | Test Result |
|---|---:|
| Test Loss | 0.8075 |
| Test Accuracy | 69.05% |
| Test F1 Score | 56.64% |

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python rumor_detection.py
```

The first run downloads the PHEME dataset and generates the processed graph
dataset. This can take significant time because BERT embeddings are computed
for the tweets.

The generated files are ignored by Git through `.gitignore`.

## Project Structure

```text
Graph-Based-Rumor-Detection/
│
├── rumor_detection.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    └── README.md
```

## Notes

The implementation follows the architecture and engineering assumptions
used in the project code, including the 712-dimensional projection before
the GAT layer and the specified dropout rates.
