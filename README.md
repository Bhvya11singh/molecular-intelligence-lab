# Molecular Intelligence Lab

Graph Neural Networks and Scientific Machine Learning for Molecular Property Prediction

## Overview

This project explores molecular machine learning using Graph Neural Networks (GNNs). Molecules are represented as graphs, where:

The long-term goal of this project is to explore Scientific Machine Learning approaches that combine machine learning, chemistry, mathematics, and physical constraints for molecular modeling.

A Graph Convolutional Network (GCN) is trained to predict molecular properties directly from graph structure.

## Dataset

* QM9 Dataset
* 130,000+ small organic molecules
* Quantum chemical properties computed using Density Functional Theory (DFT)

## Current Task

Predicting molecular dipole moment (μ) from molecular graphs.

## Technologies

* Python
* PyTorch
* PyTorch Geometric
* Graph Neural Networks
* Computational Chemistry

## Results

| Model | Property      | MAE    |
| ----- | ------------- | ------ |
| GCN   | Dipole Moment | 1.0857 |

## Experimental Results

| Model | Dataset Size | Epochs | MAE |
|---------|---------|---------|---------|
| GCN | 5000 | 10 | 1.0857 |
| GCN | 5000 | 50 | 0.9574 |
| GIN | 5000 | 50 | 0.7952 |

### Observation

GIN outperformed GCN on molecular property prediction, achieving a ~17% reduction in MAE under identical training conditions.

## Project Structure

```text
molecular-intelligence-lab/
│
├── data/
├── models/
├── notebooks/
│
├── results/
│   └── results.csv
│
├── train.py
├── evaluate.py
├── evaluate_multitask.py
├── requirements.txt
└── README.md
```

---

## Future Directions

* GIN Networks
* Multi-task Property Prediction
* Molecular Embeddings
* Scientific Machine Learning
* Molecular AI Applications

## Author

**Bhavya Singh**

IISER Mohali
Mathematics • Chemistry • AI/ML • Scientific Computing
