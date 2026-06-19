# Molecular Intelligence Lab

## Author

Bhavya Singh  
IISER Mohali

---

# Abstract

This project investigates Graph Neural Networks (GNNs) for molecular property prediction using the QM9 quantum chemistry dataset.

The study compares Graph Convolutional Networks (GCNs) and Graph Isomorphism Networks (GINs) for molecular representation learning. The project further extends to multi-task learning for simultaneous prediction of 19 quantum chemical properties.

Results demonstrate that GIN architectures outperform GCNs and that larger datasets significantly improve model performance.

---

# Introduction

Molecular property prediction is a fundamental problem in computational chemistry and drug discovery.

Traditional approaches often rely on handcrafted molecular descriptors. Recent advances in Graph Neural Networks allow molecules to be represented directly as graphs, enabling end-to-end learning from molecular structure.

This project explores:

- Graph Convolutional Networks (GCNs)
- Graph Isomorphism Networks (GINs)
- Multi-task molecular learning
- Scientific Machine Learning applications

---

# Dataset

## QM9

The QM9 dataset contains over 130,000 small organic molecules.

Each molecule includes:

- Atomic features
- Molecular graph structure
- Quantum chemical properties

Properties are computed using Density Functional Theory (DFT).

---

# Methodology

## Graph Representation

Molecules are represented as graphs:

- Nodes → Atoms
- Edges → Chemical bonds

## Models

### GCN

A Graph Convolutional Network was implemented as the baseline architecture.

### GIN

A Graph Isomorphism Network was implemented to improve molecular representation learning.

### Multi-Task Learning

The final model predicts 19 molecular properties simultaneously using a shared graph encoder and multi-output prediction head.

---

# Results

## Single Property Prediction

| Model | MAE |
|---------|---------|
| GCN (10 epochs) | 1.0857 |
| GCN (50 epochs) | 0.9574 |
| GIN (50 epochs) | 0.7952 |

## Multi-Task Learning

| Model | Validation Loss |
|---------|---------|
| MultiTask GIN (5k molecules) | 6.5868 |
| MultiTask GIN (20k molecules) | 0.3373 |

---

# Discussion

Key observations:

- GIN consistently outperformed GCN.
- Increasing dataset size improved generalization.
- Multi-task learning enabled simultaneous prediction of multiple quantum chemical properties.
- Target normalization significantly stabilized training.

---

# Future Work

Future directions include:

- Geometry-aware Graph Neural Networks
- Molecular 3D coordinate learning
- Scientific Machine Learning
- Physics-informed molecular models
- Drug discovery applications
- Materials informatics

---

# Conclusion

This project demonstrates the effectiveness of Graph Neural Networks for molecular property prediction and highlights the potential of Scientific Machine Learning approaches for computational chemistry.

The work establishes a foundation for future research in molecular AI and geometry-aware neural networks.