# A Unified Comparative Framework for Nonlinear MIMO System Identification Using Recurrent Neural Network Architectures
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--0350--4278-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0008-0350-4278)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23220104.svg)](https://doi.org/10.5281/zenodo.23220104)

## Overview

This repository contains the MATLAB code used to evaluate and compare five recurrent neural network architectures for nonlinear multi-input multi-output (MIMO) system identification:

- **LSTM** — Long Short-Term Memory
- **Elman RNN** — Elman Recurrent Neural Network
- **GRU** — Gated Recurrent Unit
- **NARX** — Nonlinear Autoregressive Network with Exogenous Inputs
- **ESN** — Echo State Network

The framework evaluates predictive accuracy, robustness to measurement noise and system delays, computational requirements, frequency-domain fidelity, and internal dynamical properties under a common experimental methodology.

The code supports the experiments reported in the associated research paper.

## Repository Structure

```text
.
├── Analysis/
│   ├── BOOT_CI.m
│   ├── BOXPLOT.m
│   ├── Generate_FFT.m
│   └── Ranking.m
├── ESN/
│   ├── ESN_delay_test.m
│   ├── ESN_noise_test.m
│   ├── ESN_parameters.m
│   ├── ESN_peak_memory.m
│   ├── ESN_stability.m
│   ├── ESN_test.m
│   └── ESN_training.m
├── Elman/
│   ├── Elman_delay_test.m
│   ├── Elman_noise_test.m
│   ├── Elman_parameters.m
│   ├── Elman_peak_memory.m
│   ├── Elman_stability.m
│   ├── Elman_test.m
│   └── Elman_training.m
├── GRU/
│   ├── GRU_delay_test.m
│   ├── GRU_noise_test.m
│   ├── GRU_parameters.m
│   ├── GRU_peak_memory.m
│   ├── GRU_stability.m
│   ├── GRU_test.m
│   └── GRU_training.m
├── LSTM/
│   ├── LSTM_delay_test.m
│   ├── LSTM_noise_test.m
│   ├── LSTM_parameters.m
│   ├── LSTM_peak_memory.m
│   ├── LSTM_stability.m
│   ├── LSTM_test.m
│   └── LSTM_training.m
└── NARX/
    ├── NARX_delay_test.m
    ├── NARX_noise_test.m
    ├── NARX_parameters.m
    ├── NARX_peak_memory.m
    ├── NARX_stability.m
    ├── NARX_test.m
    └── NARX_training.m
```

## Experimental Framework

### 1. Model Training

Each architecture has a dedicated training script:

- `*_training.m`

These scripts implement the training procedures for the corresponding network architectures.

### 2. Predictive Performance

The scripts:

- `*_test.m`

evaluate the trained models on the test dataset.

Performance is assessed using the metrics employed in the study, including mean absolute error (MAE), root mean squared error (RMSE), median absolute error (MedAE), directional accuracy (DA), and coefficient of determination (\(R^2\)).

### 3. Robustness Analysis

Two additional experiments assess the sensitivity of each architecture to disturbances:

- `*_noise_test.m` — evaluation under additive measurement noise.
- `*_delay_test.m` — evaluation under a one-step delay in the measured system response.

These tests examine model robustness without retraining the previously identified models.

### 4. Computational and Structural Analysis

The following scripts support the evaluation of model complexity and computational requirements:

- `*_parameters.m` — parameter analysis.
- `*_peak_memory.m` — peak memory usage analysis.

### 5. Internal Dynamics

The scripts:

- `*_stability.m`

evaluate the local dynamical properties of the trained predictors under the conditions defined in the corresponding implementations.

These analyses characterize model dynamics and should not be interpreted as direct assessments of the physical plant's stability or closed-loop control stability.

### 6. Statistical and Frequency-Domain Analysis

The `Analysis/` directory contains scripts for processing and comparing the experimental results:

| Script | Purpose |
|---|---|
| `BOOT_CI.m` | Bootstrap confidence interval analysis |
| `BOXPLOT.m` | Boxplot generation |
| `Generate_FFT.m` | Frequency-domain analysis |
| `Ranking.m` | Statistical ranking and comparative analysis |

## Requirements

- MATLAB.
- The MATLAB toolboxes required by the selected architecture and its implementation.
- The datasets and supporting files referenced by the scripts.

The required MATLAB release and toolbox versions depend on the functions used in each implementation.

## Usage

1. Clone or download this repository.
2. Open MATLAB and set the repository directory as the working directory.
3. Ensure that the required datasets and supporting files are available in the expected locations.
4. Run the training script for the desired architecture.
5. Execute the corresponding test, noise, delay, parameter, memory, and stability scripts as required.
6. Use the analysis scripts to process the results and generate the corresponding statistical and frequency-domain analyses.

**Important:** Check the input filenames, paths, and dependencies in each script before running it. The execution order may depend on the files generated during training and evaluation.

## Reproducibility

The repository is intended to support reproducible comparison of recurrent neural network architectures under a common experimental framework. For meaningful comparisons, use the same datasets, preprocessing procedures, evaluation conditions, and random-seed protocol specified in the associated study.

Results may vary with MATLAB versions, toolbox implementations, hardware, and numerical settings.

## Citation

If you use this code in academic work, please cite the associated research paper once its publication details are available.

## License

Please refer to the repository's `LICENSE` file for the applicable terms of use. If no license file is present, the code should not be assumed to be openly licensed for redistribution or reuse.

## Acknowledgements

This work was developed as part of research on nonlinear MIMO system identification and the comparative evaluation of recurrent neural network architectures.
