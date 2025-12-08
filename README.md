# Machine Learning for Quantum Scattering as an Inverse Problem

This repository contains a small PyTorch-based framework to study quantum scattering as an inverse problem for the 3D Schrödinger equation.  
A Lippmann–Schwinger (LS) forward solver is used to map a Yukawa-type potential to far-field scattering amplitudes, and a neural network is trained to infer the potential from synthetic data.

## Files

- **`pytorch_skeleton.py`** – Core physics/ML utilities: complex-number helpers, quadrature construction, a parametric Yukawa potential, and the `LSForward` module (differentiable LS solver for multiple wavenumbers).

- **`make_synth_data.py`** – Generates synthetic training data by fixing a “ground truth” Yukawa potential, running the LS forward solver, adding complex noise, and saving everything to a `.pt` file.

- **`train_inverse.py`** – Trains a parametric PINN to recover the Yukawa parameters from the synthetic far-field data using a complex L2 loss and light regularization.

- **`plot_results.py`** – Reloads the dataset and trained model, evaluates the learned vs. true potential and amplitudes, and produces comparison plots.

## Usage

From the project root:

```bash
# 1. Generate synthetic data
python make_synth_data.py

# 2. Train the inverse model
python train_inverse.py

# 3. Plot and compare results
python plot_results.py
