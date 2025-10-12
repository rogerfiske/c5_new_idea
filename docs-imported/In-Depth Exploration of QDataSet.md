# In-Depth Exploration of QDataSet: Quantum Datasets for Machine Learning

## Introduction
QDataSet is a comprehensive, large-scale collection of simulated quantum datasets specifically designed to advance quantum machine learning (QML). Introduced in 2021, it addresses a critical gap in the field by providing standardized, high-quality data for training, benchmarking, and testing QML algorithms, much like classical ML datasets such as MNIST or ImageNet have done for computer vision. The dataset draws from simulations of one- and two-qubit quantum systems evolving under various control pulses and noise conditions, mimicking real-world quantum engineering challenges. It supports applications in quantum control, state tomography, noise spectroscopy, and circuit synthesis, enabling both classical ML on quantum data and hybrid quantum-classical workflows.

The project stems from the QML Dataset Project, emphasizing interoperability across platforms like TensorFlow, QuTiP, and Qiskit. QDataSet is openly licensed (MIT/CC) and hosted via a GitHub repository that includes generation scripts and example notebooks. As of 2025, it remains a foundational resource for QML research, with citations in over 50 papers and integrations in benchmarks for noisy intermediate-scale quantum (NISQ) devices.

## Dataset Overview
QDataSet comprises **52 distinct datasets**, each containing **10,000 examples** (samples) derived from Monte Carlo simulations of quantum evolutions. The total compressed size is approximately **14 TB** (uncompressed exceeding 100 TB), making it one of the largest quantum-specific datasets available. Each example simulates a randomized control pulse sequence applied to an initial quantum state, evolved over time with optional noise and distortions, and measured via Pauli operators.

### Key Statistics
| Aspect | Details |
|--------|---------|
| **Total Datasets** | 52 (26 base configurations × 2 pulse shapes × 2 distortion variants) |
| **Samples per Dataset** | 10,000 examples |
| **System Dimensions** | 1-qubit: Hilbert space dim=2 (18 Pauli measurements); 2-qubit: dim=4 (52 Pauli measurements) |
| **Time Steps (M)** | 1,024 intervals (Δt) over total time T=1 |
| **Noise Realizations (K)** | Up to 2,000 Monte Carlo runs per example |
| **Storage Format** | Python Pickle files (dictionaries with NumPy arrays/tensors) in compressed ZIP archives |
| **File Sizes** | Smallest: ~1.4 GB; Largest: ~500 GB (per dataset ZIP) |
| **Generation Time** | ~3 months on HPC cluster (days for 1-qubit, weeks for 2-qubit) |

Datasets are categorized into four types based on qubit count, drift Hamiltonians, control axes, and noise profiles. This taxonomy ensures coverage of common experimental setups, such as local vs. interacting qubits.

### Dataset Categories and Naming Convention
Datasets are named using a structured format: `[PulseType]_[Qubits]q_[ControlHamiltonians]_[Noise]_[NoiseProfiles]_[_D for Distortion]`. Pulse types are Gaussian ('G') or Square ('S'). Controls and noise specify Pauli axes (e.g., 'X' for σ_x, 'IX-XI-XX' for two-qubit tensor products).

| Category | Qubits | Drift | Controls | Noise | Datasets (Examples) |
|----------|--------|-------|----------|-------|---------------------|
| **1** | 1 | Z | X | Z | 10 (e.g., G_1q_X_N1, S_1q_X_N0_D) – Basic single-axis noise (N1-N4 + noiseless N0) |
| **2** | 1 | Z | XY (X+Y) | XZ | 8 (e.g., G_1q_XY_XZ_N1-N5, S_1q_XY_XZ_N0) – Multi-axis noise pairs (e.g., N1+N5, N3+N6) |
| **3** | 2 | Z1, 1Z | X1, 1X | Z1, 1Z | 4 (e.g., G_2q_X1-1X_Z1-1Z_N1-N6) – Local correlated noise on separable controls |
| **4** | 2 | Z1, 1Z | X1, 1X, XX | Z1, 1Z | 6 (e.g., S_2q_IX-XI-XX_IZ-ZI_N1-N5_D) – Interacting controls with local noise |

Noise profiles (N0-N6) emulate experimental imperfections:
- **N0**: Noiseless baseline.
- **N1/N5**: 1/f-like frequency noise with Gaussian PSD bumps.
- **N2**: Stationary Gaussian colored noise (convolution-based).
- **N3**: Non-stationary Gaussian colored noise (multiplicative).
- **N4**: Non-Gaussian, non-stationary (squared Gaussian).
- **N6**: Correlated to others (e.g., squared N1 for β²(t)).

Distortions ('_D') apply a Chebyshev low-pass filter to pulses, simulating hardware filtering effects.

## Generation Process
QDataSet was generated using custom Python scripts (available in the repository's `simulation` folder) on a high-performance computing (HPC) cluster at the University of Technology Sydney. Simulations employ exact Monte Carlo methods to evolve quantum states via the Schrödinger equation, avoiding approximations like Lindblad master equations for higher fidelity.

### Step-by-Step Process
1. **Parameter Sampling**: For each of 10,000 examples, randomly generate pulse sequences (5 non-overlapping pulses per example; amplitudes uniform in [-100, 100]; Gaussian σ = T/(12M)).
2. **Hamiltonian Construction**: 
   - Drift (H₀): Static, e.g., (1/2)Ω σ_z (Ω=12 for 1-qubit).
   - Control (H₁(t)): Time-dependent, f(t) ⊗ σ_α (α=X/Y/Z).
   - Noise (Hₙ(t)): Stochastic β(t) ⊗ σ_β, with β drawn per profile (e.g., convolve white noise for coloring).
3. **Evolution Simulation**: Discretize time into M=1024 steps; compute unitary U(T) ≈ ∏ exp(-i H(jΔt) Δt) using Suzuki-Trotter. For noise: U_total = Ũ_interact U_noiseless over K=2,000 realizations.
4. **Measurement**: Prepare 6 initial Pauli eigenstates; evolve ρ(t) = U(t) ρ(0) U†(t); compute <O> = Tr(ρ O) for Pauli O (18/52 values total). Average for E_O; per-realization for V_O (noise-encoded operator).
5. **Distortion (if '_D')**: Filter pulses with Chebyshev analog (order 5, cutoff 0.1Ω).
6. **Batching and Storage**: Process in batches of 50; serialize as Pickle dicts (keys: `simulation_parameters`, `pulses`, `expectations`, `noise`, `H0`, `U0`, etc.); ZIP compress.

Dependencies: Python 3, NumPy (for tensors), TensorFlow 2.5.0 (optional for verification). Two-qubit runs required 360 GB RAM and GPU acceleration. Scripts like `dataset_G_1q_X.py` allow regeneration.

## Structure and Contents
Each dataset ZIP contains ~200 Pickle files (50 examples/batch × 200 batches), where each Pickle is a dictionary of NumPy arrays/tensors. Core keys include:
- **simulation_parameters**: Dict with dim, Ω, operators (static/dynamic/noise/measurement as 2x2/4x4 matrices), initial_states (vectors), T=1, num_ex=10,000, K=2,000, noise_profile, pulse_shape.
- **pulse_parameters**: Arrays of amplitudes, means/σ for Gaussians.
- **time_range**: [0, Δt, 2Δt, ..., T] (shape: M+1).
- **pulses/distorted_pulses**: Waveforms (shape: [num_pulses, M]).
- **expectations**: Pauli <O>(T) (shape: [num_states, 3] for 1-qubit; higher for 2-qubit).
- **noise**: H_n(t) realizations (shape: [K, M, dim, dim]).
- **H0/H1/UI/U0**: Hamiltonians/unitaries (various shapes, e.g., [M, dim, dim]).
- **V0/E0**: Noise-averaged operators/expectations (shape: [3, K] per observable).

Data is low-dimensional (qubit-focused) but high-volume, with sparsity in noise/pulses. Preprocessing suggestions: SVD for dimensionality reduction, smoothing for time series.

## Applications in Quantum Machine Learning
QDataSet excels in "classical ML for quantum data" paradigms, training models to optimize quantum systems without full quantum access. Key tasks:

- **Quantum State Tomography**: Reconstruct ρ(T) from partial measurements {E_O}. Input: Subset of expectations; Label: True ρ(T); Metric: Fidelity F(ρ, ˆρ) ∈ [0,1]. ML: Neural nets reduce measurements from O(d^4) to O(d^2).
- **Noise Spectroscopy**: Estimate noise operators {V_O} from controls/observables. Input: Pulses; Label: {E_O}; Metric: MSE(E_O, ˆ{E_O}). Enables greybox models encoding physics.
- **Quantum Control/Circuit Synthesis**: Optimize pulses to achieve target U_T or ρ(T). Input: H₀/H₁; Label: ρ(T) or intermediates ρ(t_j); Metric: Average fidelity or trace distance D(ρ,σ)=½ Tr|ρ-σ|. Supports dynamic decoupling for noise mitigation.

Workflow: Split data (80% train, 10% val, 10% test); use supervised regression (e.g., XGBoost for pulses) or unsupervised clustering (e.g., k-means on V_O). Hybrid extensions: Feed ML-optimized pulses to variational quantum circuits.

## Benchmarks and Evaluations
While the original paper focuses on design over exhaustive benchmarks, QDataSet supports standardized QML evaluations:
- **Metrics**: Fidelity (state/operator), MSE/RMSE (expectations/pulses), quantum relative entropy S(ρ‖σ), batch fidelity (averaged over examples).
- **Protocols**: Supervised (labeled tomography), unsupervised (noise clustering), semi-supervised (partial measurements). Compare architectures: Linear (GLM), neural (MLP/VQC), ensembles (XGBoost).
- **Challenges Addressed**: Barren plateaus via PCA; overfitting via out-of-sample noise splits; scalability (subsample K for faster runs).

Repository notebooks demonstrate baselines, e.g., fidelity >0.95 for control tasks with neural nets. Community benchmarks (post-2021) report AUC~0.85-0.92 for noise classification on subsets.

## Access and Usage
- **Download**: Via Cloudstor (https://cloudstor.aarnet.edu.au/plus/s/rxYKXBS7Tq0kB8o) – one ZIP per dataset. GitHub: https://github.com/eperrier/QDataSet (examples/simulation folders).
- **Requirements**: Python 3+; NumPy/Pickle (built-in). No additional installs needed for loading.
- **Example Code** (Loading and Basic Analysis):
  ```python:disable-run
  import pickle
  import numpy as np
  import zipfile  # For extraction if needed

  # Assume ZIP extracted to './G_1q_X/'
  with open('./G_1q_X/dataset_batch_0.pkl', 'rb') as f:  # First batch
      data = pickle.load(f)

  # Access data
  params = data['simulation_parameters']  # Dict: {'dim': 2, 'noise_profile': 'N0', ...}
  expectations = data['expectations']    # Shape: (6 states, 3 Paulis), values in [-1,1]
  pulses = data['pulses']                # Shape: (5 pulses, 1024 time steps)

  # Quick stat: Average fidelity proxy (e.g., variance in expectations)
  print(np.mean(expectations, axis=0))   # [x, y, z] expectations
  print(np.var(pulses))                  # Pulse variability
  ```
  Full notebooks in `/examples` cover ML pipelines (e.g., scikit-learn regression on pulses).

## Related Resources and Future Directions
- **Paper**: arXiv:2108.06661 (full PDF details appendices on noise/math).
- **Citations/Publications**: Published in Scientific Data (2022); influences works on VQE datasets and NISQ benchmarks.
- **Extensions**: Community calls for higher-qubit versions or experimental data integrations. As of 2025, forks explore 3-qubit extensions for error correction tasks.

QDataSet's uniform, noisy simulations make it ideal for robust QML development, bridging theory and NISQ hardware. For hands-on exploration, start with 1-qubit noiseless datasets.
```