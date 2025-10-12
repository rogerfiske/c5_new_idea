# QuTiP: Quantum Toolbox in Python for Simulations

## Overview
QuTiP (Quantum Toolbox in Python) is an open-source software library designed for simulating the dynamics of open quantum systems. It provides efficient numerical tools for modeling a wide range of quantum phenomena, including closed and open systems, with support for arbitrary time-dependent Hamiltonians. Built on NumPy, SciPy, and Cython for high-performance computations, and Matplotlib for visualization, QuTiP is widely used in research areas such as quantum optics, quantum information, trapped ions, superconducting circuits, quantum nanomechanical resonators, and optomechanics. It is cross-platform (Linux, macOS, Windows) and freely available under the BSD license, with a vibrant community contributing via GitHub. As of October 2025, QuTiP has been downloaded over a million times and is employed at universities, labs, and quantum tech companies worldwide.

The library excels in user-friendly syntax, allowing complex simulations with minimal code, making it suitable for both education and advanced research. It supports exact solutions for small systems and approximate methods (e.g., master equation solvers) for larger ones, handling noise, dissipation, and correlations.

## Key Features
- **Quantum Objects (Qobj)**: Unified representation for states (kets, bras, density matrices) and operators, with automatic type checking and mathematical operations.
- **Time Evolution Solvers**: Master equation (mesolve), Monte Carlo (mcsolve), floquet methods, and stochastic solvers for open systems.
- **Visualization Tools**: Bloch sphere plotting, Wigner functions, expectation value trajectories, and circuit diagrams.
- **Advanced Modules**: Continuous variables, permutation-invariant quantum states (piqs), quantum control (pulse optimization), and tomography.
- **Parallelization**: Support for MPI and multiprocessing via optional backends.
- **Extensibility**: Easy integration with Jupyter notebooks for interactive simulations.

Recent updates in version 5.x include improved environment handling for baths, better parallel map backends, and enhancements to the Drude-Lorentz-Pade bath model for non-Markovian dynamics.

## Latest Version
The current stable release is QuTiP 5.2.1, released on August 27, 2025. This version includes bug fixes, performance optimizations for Cython-compiled functions, and expanded documentation for new solvers. The development roadmap outlines plans for version 6.0, focusing on GPU acceleration and larger-scale simulations.

## Installation
QuTiP requires Python 3.9+ (or 3.6+ for older versions). Core dependencies are NumPy (1.22+ <2.0) and SciPy (1.8+). Optional packages enhance functionality: Matplotlib for plotting, Cython for time-dependent Hamiltonians (requires a C++ compiler), and others like cvxpy for norms or pytest for testing.

Use virtual environments (e.g., venv or conda) to avoid conflicts.

### Quick Start with pip
```
pip install qutip
```
This installs core dependencies only.

### With Conda (Recommended for Beginners)
Add the conda-forge channel:
```
conda config --append channels conda-forge
```
Then install:
```
conda install qutip
```
For a new environment:
```
conda create -n qutip-env python qutip matplotlib jupyter
conda activate qutip-env
```

### From Source (For Developers)
Download from GitHub or PyPI. Install build tools:
```
pip install setuptools wheel packaging cython 'numpy<2.0.0' scipy
```
Then:
```
pip install .
```
For editable development:
```
pip install -e .
```
On Windows, use Visual Studio Community (free) with C++ build tools and Windows SDK. Launch from the Developer Command Prompt.

### Verification
After installation:
```python
import qutip as qt
qt.about()  # Displays versions and system info
```
Run tests (requires pytest):
```
pytest qutip/qutip/tests
```
This takes 10-30 minutes; expect some skips for optional features.

Troubleshooting: Ensure no system Python pollution; check for C++ compiler on Windows. If MPI tests hang, update mpi4py.

## Basic Usage Examples
QuTiP revolves around the `Qobj` class for quantum objects. Below are foundational examples for simulations. All code assumes `from qutip import *; import numpy as np; import matplotlib.pyplot as plt`.

### 1. Creating States and Operators
- Fock state (e.g., |3⟩ in 5-level system):
  ```python
  fock = basis(5, 3)
  print(fock)
  ```
  Output: A ket with 1 at position 3.

- Coherent state density matrix:
  ```python
  rho = coherent_dm(5, 1)  # α=1 in 5-level truncated space
  print(rho.diag())  # Diagonal (populations): [0.3679, 0.3676, 0.1852, 0.0581, 0.0212]
  ```

- Pauli-Z operator:
  ```python
  sz = sigmaz()
  print(sz)  # [[1, 0], [0, -1]]
  ```

- Custom operator from NumPy:
  ```python
  rand_op = Qobj(np.random.rand(4, 4))
  ```

### 2. Time Evolution
Simulate a qubit under Hamiltonian H = σ_z / 2, starting from |+⟩ state.
```python
H = sigmaz() / 2  # Hamiltonian
psi0 = (basis(2, 0) + basis(2, 1)).unit()  # Initial |+> state
times = np.linspace(0, 10, 100)
result = mesolve(H, psi0, times, [], [sigmaz()])  # Expectation of σ_z
plt.plot(times, result.expect[0])
plt.xlabel('Time'); plt.ylabel('<σ_z>'); plt.show()
```
This plots oscillatory expectation values, demonstrating Rabi-like dynamics.

For open systems (e.g., with decoherence):
```python
c_ops = [np.sqrt(0.1) * sigmam()]  # Collapse operator for decay
result = mesolve(H, psi0, times, c_ops, [sigmaz()])
```

### 3. Plotting
- Bloch sphere for a state:
  ```python
  b = Bloch()
  b.add_states(psi0)
  b.show()
  ```

- Wigner function for a coherent state:
  ```python
  xvec = np.linspace(-10, 10, 200)
  W = wigner(Qobj(coherent(10, 2)), xvec, xvec)
  plt.contourf(xvec, xvec, W, 100); plt.colorbar(); plt.show()
  ```

For more, explore tutorials at qutip.org/qutip-tutorials, including Jupyter notebooks for full simulations.

QuTiP's simplicity and power make it ideal for prototyping quantum algorithms or analyzing experimental data. For API details, see the full docs.