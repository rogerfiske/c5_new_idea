# Quantum “Database” Formats — What Exists Today

> Short answer: There is **no single, standardized “quantum database format.”** In practice, teams store *classical* artifacts that describe quantum programs, results, and hardware signals using a mix of open quantum program formats plus conventional data containers (Parquet/CSV/HDF5/JSON). “Quantum databases” in the literal sense typically refer to **QRAM** *hardware concepts*, not a file format.

---

## 1) Quantum **Program / Circuit** Formats (text or IR)
- **OpenQASM 3** — widely used, vendor‑neutral circuit language with classical control, timing, and subroutines.  *Files*: `.qasm`
- **QIR (Quantum Intermediate Representation)** — LLVM‑based IR for cross‑toolchain exchange.  *Files*: LLVM bitcode or textual IR (`.bc`, `.ll`)
- **Quil / Quil‑T (Rigetti)** — instruction language; Quil‑T adds pulse/analog timing.  *Files*: `.quil`
- **Blackbird (Xanadu, CV photonics)** — assembly for continuous‑variable programs.  *Files*: `.bb`

## 2) **Pulse‑level** Control Formats
- **OpenPulse** (originally JSON wire format; now also a grammar within OpenQASM docs) for low‑level drive/readout waveforms.  *Files*: JSON payloads or textual pulse specs

## 3) **Serialization** / Exchange of Circuits & Results
- **Qiskit QPY** — binary, versioned format for portable circuit graphs and metadata.  *Files*: `.qpy` (optionally gzip‑compressed)
- **Qobj** (historical Qiskit “quantum object”) — standardized experiment payloads (JSON).  *Files*: JSON

## 4) **Result & State Data (Classical Containers)**
Use standard, scalable formats with clear schemas:
- **Parquet** — columnar, compressed; great for large result sets.
- **HDF5** — hierarchical arrays (good for statevectors, density matrices, tomography data).
- **Arrow / Feather** — fast interchange between Python/Rust/Julia.
- **CSV/JSON** — simple logs, configuration, and small result tables.

### Common state encodings (for simulators/analysis)
- **Statevector**: complex amplitude vector (save as NumPy `.npy`, HDF5 dataset, or Parquet columns real/imag).
- **Density matrix**: complex square matrix (HDF5 dataset or Parquet with structured fields).
- **Measurement shots**: counts histogram, bitstrings table (Parquet/CSV).

## 5) “Quantum Databases” in the literal sense (QRAM)
- **QRAM** is a *hardware architecture* for addressing superpositions of memory cells. It is **not a file format** and is not broadly available in production hardware. When people say “quantum database,” they usually mean algorithms (e.g., Grover search) that **query** a dataset, not a new on‑disk format.

---

## Practical Starter Schema (what to store today)
- `/program/` — OpenQASM 3 (`.qasm`) and/or QIR (`.ll`/`.bc`) snapshots.
- `/circuit/` — QPY (`.qpy`) for lossless round‑trips across Qiskit.
- `/pulse/` — OpenPulse JSON packets (if doing pulse‑level experiments).
- `/results/` — Parquet tables for shots, counts, metadata (job id, backend, seed).
- `/states/` — HDF5 groups for `statevector` or `density_matrix` (complex64/complex128).
- `/config/` — JSON for run parameters, transpilation passes, seeds, calibration hashes.
- `/provenance/` — Git SHA, SDK versions, backend version, timestamp.

This combo works across IBM Qiskit, Rigetti, Cirq/Cirq‑IonQ, and photonic stacks while keeping the heavy data in efficient classical containers.

---

## FAQ
- **Is there a single “quantum DB” file I should use?** No. Pair an open *program format* (OpenQASM 3 or QIR) with standard *data containers* (Parquet/HDF5).- **Can I store a qubit state “as is”?** Not directly; you store **classical descriptions** (amplitudes, density matrices, tomography reconstructions).- **Future‑proof?** Keep *both* a human‑readable program (OpenQASM 3) and a compiler‑friendly IR (QIR/QPY). Version everything in metadata.

---

### Pointers (see chat for source links)
OpenQASM 3 | QIR | Quil / Quil‑T | Blackbird | OpenPulse | QPY
