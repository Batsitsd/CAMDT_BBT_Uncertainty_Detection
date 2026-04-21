# CAMDT-BBT: Uncertainty-Aware Underwater Cable Fault Detection

## Overview

This repository provides an implementation of the **CAMDT-BBT framework**, a Bayesian-enhanced Transformer architecture designed for robust and uncertainty-aware underwater cable fault detection.

The method integrates multi-scale feature learning and probabilistic modeling to improve both detection performance and uncertainty quantification under complex underwater environments.

---

## Relation to the Paper

This implementation corresponds to the method described in our paper:

**"Bayesian Transformer for Robust Underwater Cable Detection"**

The code structure follows the main components presented in Section 3 of the paper:

* **CAMDT (Channel-Adaptive Multi-scale Dual Transformer)** – Feature extraction and multi-scale representation
* **AMDT (Attention-guided Multi-scale Dual Transformer)** – Attention-based feature refinement
* **BBT (Bayesian Boosting Transformer)** – Probabilistic modeling and uncertainty-aware prediction

These modules are implemented in a modular way to reflect the architectural design and logical flow described in the manuscript.

---

## Method Overview

The overall framework consists of three stages:

1. **Multi-scale feature extraction**
   A transformer-based backbone captures hierarchical representations from input images.

2. **Attention-guided feature interaction**
   Cross-scale information is refined using attention mechanisms.

3. **Bayesian-enhanced transformer modeling**
   Stochasticity is introduced into the attention projections to enable uncertainty estimation.

---

## Uncertainty Estimation

Uncertainty is modeled using a Monte Carlo sampling strategy:

* Dropout is applied to key projection layers (Q/K/V)
* Multiple stochastic forward passes approximate posterior sampling
* Predictive mean and variance are computed as uncertainty measures

This design enables the model to capture both prediction confidence and variability.

---

## Code Structure

```bash
models/
├── camdt.py        # Multi-scale feature extraction module
├── amdt.py         # Attention-guided feature refinement
├── bbt.py          # Bayesian transformer module
└── model.py        # Integrated CAMDT-BBT architecture

utils/
└── uncertainty.py  # Monte Carlo uncertainty estimation
```

---

## Reproducibility

This repository provides a structured implementation of the proposed method with a focus on clarity and reproducibility of the core ideas.

Due to dataset availability and engineering dependencies, certain components (e.g., full training pipeline and dataset preprocessing) are not included. However, the implementation preserves the essential architectural design and methodological details described in the paper.

Key experimental settings used in this implementation:

* Input resolution: 256 × 256
* Number of stochastic forward passes: T = 20
* Dropout applied in attention projection layers

These settings are consistent with the methodology described in the manuscript and enable partial reproduction of the reported behavior.

---

## Notes

* The implementation is organized to clearly reflect the model design described in the paper
* Some engineering-level optimizations are omitted for clarity
* The code is intended to facilitate understanding and further research

---

```
