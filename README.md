# Molecular Intelligence Lab

Graph Neural Networks for Molecular Property Prediction using the QM9 Quantum Chemistry Dataset.

## Overview

This project explores molecular machine learning using Graph Neural Networks (GNNs). Molecules are represented as graphs, where:

* Nodes represent atoms
* Edges represent chemical bonds

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

## Project Structure

```text
molecular-intelligence-lab/
├── data/
├── models/
├── notebooks/
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Future Work

* GIN Networks
* Multi-task Property Prediction
* Molecular Embeddings
* Scientific Machine Learning
* Molecular AI Applications

## Author

Bhavya Singh

IISER Mohali
Mathematics • Chemistry • AI/ML • Scientific Computing
