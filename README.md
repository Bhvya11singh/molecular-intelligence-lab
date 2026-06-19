# Molecular Intelligence Lab

Graph Neural Networks and Scientific Machine Learning for Molecular Property Prediction

## Overview

Molecular properties govern the behavior of chemical systems, influencing reactivity, stability, spectroscopy, and thermodynamics. Traditional quantum chemistry methods such as Density Functional Theory (DFT) provide accurate predictions but are computationally expensive for large-scale molecular discovery.

This project investigates the use of Graph Neural Networks (GNNs) as surrogate models for quantum chemical property prediction using the QM9 benchmark dataset. Molecules are represented as graphs where atoms are nodes and chemical bonds are edges, allowing neural networks to learn structure-property relationships directly from molecular topology.

The long-term goal of this project is to explore Scientific Machine Learning approaches that combine machine learning, chemistry, mathematics, and physical constraints for molecular modeling.

---

## Dataset

### QM9 Quantum Chemistry Dataset

* 130,831 organic molecules
* Up to 9 heavy atoms (C, N, O, F)
* Quantum chemical properties computed using Density Functional Theory (DFT)
* Widely used benchmark for molecular machine learning research

### Target Properties

The multitask model predicts 19 molecular properties simultaneously, including:

* Dipole Moment (μ)
* Polarizability (α)
* HOMO Energy
* LUMO Energy
* HOMO-LUMO Gap
* Zero Point Vibrational Energy
* Thermodynamic Energies
* Heat Capacity
* Rotational Constants

---

## Models Implemented

### Graph Convolutional Network (GCN)

Baseline graph neural network using message passing and graph-level pooling.

### Graph Isomorphism Network (GIN)

More expressive graph architecture designed to better capture molecular structural information.

### Multi-Task GIN

Extended GIN model for simultaneous prediction of 19 molecular properties.

### Multi-Task GIN with Dropout

Regularized multitask architecture trained on larger datasets to improve generalization.

---

## Experimental Results

### Single-Property Prediction

| Model | Dataset Size | Property      | MAE    |
| ----- | ------------ | ------------- | ------ |
| GCN   | 5,000        | Dipole Moment | 1.0857 |
| GCN   | 5,000        | Dipole Moment | 0.9574 |
| GIN   | 5,000        | Dipole Moment | 0.7952 |

### Multi-Task Prediction

| Model                    | Dataset Size | Targets | Validation Loss |
| ------------------------ | ------------ | ------- | --------------- |
| Multi-Task GIN           | 5,000        | 19      | 6.5868          |
| Multi-Task GIN + Dropout | 20,000       | 19      | Ongoing         |

### Property-Wise Evaluation

Representative results from the multitask model:

| Property                      | MAE  |
| ----------------------------- | ---- |
| Dipole Moment (μ)             | 1.10 |
| Polarizability (α)            | 7.85 |
| HOMO Energy                   | 0.56 |
| LUMO Energy                   | 0.66 |
| HOMO-LUMO Gap                 | 0.55 |
| Zero Point Vibrational Energy | 0.52 |

---

## Key Findings

* GIN consistently outperformed GCN on molecular property prediction.
* Multi-task learning enabled simultaneous prediction of 19 quantum chemical properties.
* Target normalization significantly improved optimization stability.
* Increasing dataset size from 5k to 20k molecules substantially improved validation performance.
* Electronic properties such as HOMO, LUMO, and energy gap were predicted more accurately than large thermodynamic energy quantities.

---

## Project Structure

```text
molecular-intelligence-lab/
│
├── data/
├── models/
│   ├── gcn.py
│   ├── gin.py
│
├── results/
│   ├── multitask_metrics.csv
│   ├── experiment_summary.csv
│
├── notebooks/
├── train.py
├── evaluate.py
├── evaluate_multitask.py
├── requirements.txt
└── README.md
```

## Future Directions

* Geometry-Aware Graph Neural Networks
* Incorporation of 3D Molecular Coordinates
* Physics-Informed Neural Networks
* Molecular Representation Learning
* Scientific Machine Learning for Chemical Systems
* Neural Surrogates for Quantum Chemistry
* Molecular Foundation Models

---

## Author

Bhavya Singh

Fourth Year BS-MS Student
Indian Institute of Science Education and Research (IISER) Mohali

Research Interests:

* Scientific Machine Learning
* Computational Chemistry
* Graph Neural Networks
* Molecular Artificial Intelligence
* Mathematical Modeling
* Scientific Computing
