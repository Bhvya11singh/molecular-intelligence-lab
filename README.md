# Molecular Intelligence Lab

Graph Neural Networks for Molecular Property Prediction using the QM9 Quantum Chemistry Dataset.

## Overview

Molecular properties are fundamentally determined by atomic composition and chemical structure. This project explores how Graph Neural Networks (GNNs) can learn molecular representations directly from molecular graphs and predict quantum-chemical properties.

In molecular graphs:

* Nodes represent atoms
* Edges represent chemical bonds

The goal is to predict molecular properties directly from graph structure without manually engineered features.

---

## Dataset

### QM9 Quantum Chemistry Dataset

* 130,000+ small organic molecules
* Quantum chemical properties computed using Density Functional Theory (DFT)
* Widely used benchmark dataset in molecular machine learning

Dataset source: QM9

---

## Objective

Predict the molecular dipole moment (μ) using Graph Neural Networks.

This serves as a foundational task in molecular machine learning and computational chemistry.

---

## Models Implemented

### Graph Convolutional Network (GCN)

* Graph convolution layers
* Global mean pooling
* Fully connected prediction head

### Graph Isomorphism Network (GIN)

* GINConv layers
* Learnable aggregation functions
* Improved expressive power for graph representation learning

---

## Technologies

* Python
* PyTorch
* PyTorch Geometric
* Graph Neural Networks
* Computational Chemistry
* Scientific Machine Learning

---

## Experimental Results

| Model | Dataset Size | Epochs | MAE    |
| ----- | ------------ | ------ | ------ |
| GCN   | 5000         | 10     | 1.0857 |
| GCN   | 5000         | 50     | 0.9574 |
| GIN   | 5000         | 50     | 0.7952 |

### Key Observation

Under identical training conditions, the Graph Isomorphism Network (GIN) achieved approximately **17% lower Mean Absolute Error (MAE)** than the Graph Convolutional Network (GCN), demonstrating improved molecular representation learning on the QM9 dataset.

---

## Project Structure

```text
molecular-intelligence-lab/
│
├── data/
├── models/
│   ├── gcn.py
│   ├── gin.py
│   └── *.pth
│
├── notebooks/
│
├── results/
│   └── results.csv
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## Future Work

* Multi-property prediction across all QM9 targets
* Message Passing Neural Networks (MPNN)
* Molecular embeddings and representation learning
* Physics-informed graph neural networks
* Molecular generation and inverse design
* Scientific Machine Learning for chemical systems

---

## Research Areas

This project lies at the intersection of:

* Computational Chemistry
* Graph Theory
* Machine Learning
* Scientific Computing
* Artificial Intelligence for Science

---

## Author

**Bhavya Singh**

IISER Mohali

Mathematics • Chemistry • Machine Learning • Scientific Computing
