# Molecular Intelligence Lab

Graph Neural Networks, Scientific Machine Learning, and Molecular AI for Quantum Chemistry

## Overview

Molecular Intelligence Lab is a research-oriented project exploring the intersection of:

* Mathematics
* Chemistry
* Artificial Intelligence
* Scientific Machine Learning
* Computational Chemistry

The project investigates how Graph Neural Networks (GNNs) can learn molecular representations directly from molecular graph structures and predict quantum chemical properties.

Molecules are represented as graphs where:

* Nodes represent atoms
* Edges represent chemical bonds

The long-term objective is to build physically informed machine learning models that combine data-driven learning with scientific and chemical knowledge for molecular modeling and discovery.

---

## Dataset

### QM9 Quantum Chemistry Dataset

* 130,000+ small organic molecules
* Computed using Density Functional Theory (DFT)
* 19 quantum chemical target properties
* Standard benchmark dataset for molecular machine learning

Properties include:

* Dipole Moment (μ)
* Polarizability (α)
* HOMO Energy
* LUMO Energy
* HOMO-LUMO Gap
* Internal Energy
* Heat Capacity
* Rotational Constants
* And additional quantum chemical descriptors

---

## Research Objectives

### Phase 1

Single-property molecular prediction using Graph Convolutional Networks (GCN).

### Phase 2

Improved molecular representation learning using Graph Isomorphism Networks (GIN).

### Phase 3

Multi-task molecular learning for simultaneous prediction of 19 quantum chemical properties.

### Phase 4 (Ongoing)

Scientific Machine Learning and geometry-aware molecular neural networks using molecular 3D coordinates.

---

## Technologies

* Python
* PyTorch
* PyTorch Geometric
* Graph Neural Networks
* Scientific Machine Learning
* Computational Chemistry
* Quantum Chemistry Datasets
* Data Analysis & Visualization

---

## Experimental Results

### Single Property Prediction

| Model | Dataset Size | Epochs | Target        | Performance  |
| ----- | ------------ | ------ | ------------- | ------------ |
| GCN   | 5,000        | 10     | Dipole Moment | MAE = 1.0857 |
| GCN   | 5,000        | 50     | Dipole Moment | MAE = 0.9574 |
| GIN   | 5,000        | 50     | Dipole Moment | MAE = 0.7952 |

### Multi-Task Molecular Property Prediction

| Model                                       | Dataset Size | Targets       | Performance              |
| ------------------------------------------- | ------------ | ------------- | ------------------------ |
| Multi-Task GIN                              | 5,000        | 19 Properties | Validation Loss = 6.5868 |
| Multi-Task GIN (3 Layers, 128 Hidden Units) | 20,000       | 19 Properties | Validation Loss ≈ 0.34   |

## Advanced Multi-Task Molecular Prediction

Implemented an enhanced Residual Graph Isomorphism Network (GIN) featuring:

- 3 GIN layers
- Batch Normalization
- Residual Connections
- Dropout Regularization
- Adaptive Learning Rate Scheduling

The model was trained on 20,000 QM9 molecules to jointly predict 19 quantum chemical properties.

### Best Result

| Model | Dataset Size | Targets | Best Validation Loss |
|--------|-------------|---------|---------------------|
| Residual Multi-Task GIN | 20,000 | 19 | 0.2424 |

 
### Property-wise Results (20,000 Molecules) 
| Property  | MAE     |
| --------- | ------- |
| mu        | 0.906   |
| alpha     | 2.917   |
| homo      | 0.502   |
| lumo      | 0.472   |
| gap       | 0.525   |
| r2        | 91.170  |
| zpve      | 0.217   |
| u0        | 338.871 |
| u298      | 339.279 |
| h298      | 339.407 |
| g298      | 338.983 |
| cv        | 2.140   |
| u0_atom   | 2.868   |
| u298_atom | 2.881   |
| h298_atom | 2.892   |
| g298_atom | 2.705   |
| a         | 194.512 |
| b         | 0.337   |
| c         | 0.224   |

## Molecular Representation

### 3D Molecular Structure

<p align="center">
  <img src="assets/ethanol.png" width="400">
</p>

### Molecules as Graphs

<p align="center">
  <img src="assets/molecular_graph.png" width="400">
</p>

### Residual Multi-Task GIN Architecture

<p align="center">
  <img src="assets/architecture_diagram.png" width="500">
</p>

### Training Dynamics

<p align="center">
  <img src="assets/training_curve.png" width="500">
</p>

## Key Findings

* GIN consistently outperformed GCN for molecular property prediction.
* Increasing model depth and hidden dimensions significantly improved performance.
* Multi-task learning enabled simultaneous prediction of 19 quantum chemical properties.
* Scaling from 5,000 to 20,000 molecules substantially improved model generalization.
* Architecture improvements produced larger gains than simple regularization techniques.

---

## Project Structure

```text
molecular-intelligence-lab/
│
├── data/
├── models/
│   ├── gcn.py
│   └── gin.py
│
├── notebooks/
│
├── results/
│   ├── training_curve.png
│   ├── experiment_summary.csv
│   ├── multitask_metrics.csv
│   └── multitask_metrics_20k.csv
│
├── train.py
├── evaluate.py
├── evaluate_multitask.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Bhvya11singh/molecular-intelligence-lab
cd molecular-intelligence-lab

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

Single-task evaluation:

```bash
python evaluate.py
```

Multi-task evaluation:

```bash
python evaluate_multitask.py
```

---

## Future Directions

* Geometry-Aware Molecular Neural Networks
* Molecular 3D Coordinate Learning
* SchNet-inspired Architectures
* DimeNet-style Message Passing
* Scientific Machine Learning
* Physics-Informed Neural Networks
* Molecular Representation Learning
* Drug Discovery Applications
* Materials Informatics

---

## Installation

```bash
git clone https://github.com/Bhvya11singh/molecular-intelligence-lab
cd molecular-intelligence-lab

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Evaluation

```bash
python evaluate.py
```

## Author

**Bhavya Singh**

IISER Mohali

Mathematics • Chemistry • Artificial Intelligence • Scientific Computing • Molecular Machine Learning

---

## Research Areas

Computational Chemistry • Graph Neural Networks • Molecular AI • Scientific Machine Learning • Applied Mathematics • Quantum Chemistry
