# Contributing to Fly Connectome Research

Thank you for your interest in contributing to the **Fly Connectome Research** project!

This repository investigates network dynamics, recurrent recruitment, and structural lesion vulnerability in the *Drosophila melanogaster* connectome using empirical electron-microscopy wiring diagrams and Leaky Integrate-and-Fire (LIF) models.

---

## 🔒 Scientific Freeze Policy (Phases 1–7)

> [!IMPORTANT]
> **Phases 1 through 7 are completed, validated, and locked.**

To protect scientific integrity and reproducibility:
- **Do NOT modify** existing numerical simulation results in `results/tables/`.
- **Do NOT alter** existing publication figures in `results/figures/`.
- **Do NOT modify** baseline model parameters or deterministic seeds used in Phases 1–7.
- **Do NOT alter** governing equations or subnetwork selection logic for historical runs.

If you believe you have discovered a genuine bug, numerical error, or reproducibility discrepancy in Phases 1–7:
1. Open an Issue with a minimal reproduction script.
2. Clearly explain the discrepancy without modifying historical CSV outputs in a PR.

---

## 🎯 Planned Contributions (Phase 8 & Beyond)

All new experimental work must align with the formal research plan codified in [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md):

- **Phase 8 (Topological Determinants of Lesion Sensitivity)**:
  - Metric extraction scripts for in-degree, out-degree, total synaptic weight, PageRank, and betweenness.
  - Stratified degree-matched null model generators.
  - Benjamini–Hochberg False Discovery Rate (FDR) statistical procedures.
- **Phase 9 (Density and Recurrence Scaling)**:
  - Subnetwork extraction across intermediate densities ($\rho \in [0.01, 0.11]$).
  - Degree-preserving edge-swap null algorithms (Maslov–Sneppen rewiring).
- **Phase 10 (ROI-Specific Perturbations)**:
  - Compartment filtering for Mushroom Body (MB), Central Complex (CX), and Antennal Lobe (AL).
- **Model Extensions**:
  - Excitatory/inhibitory neurotransmitter mapping from FlyWire annotations.
  - Conductance-based kinetic implementations and synaptic delays.
  - GPU tensor integration via PyTorch sparse tensors for whole-brain scaling.

---

## 🛠 Development Setup

### 1. Prerequisites
- **Python**: Version `3.13` or higher.
- **Operating System**: Windows, macOS, or Linux.
- **Hardware**: Standard laptop CPU with $\ge 8\,\text{GB}$ RAM (no GPU required).

### 2. Setting Up the Environment

```bash
# Clone the repository
git clone https://github.com/alok-108/fly-connectome-research.git
cd fly-connectome-research

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Verify Existing Tests
Before making any changes, ensure all 64 existing unit tests pass:
```bash
pytest -v
```

---

## 📐 Scientific & Computational Invariants

All new contributions must adhere to the following core architectural rules:

### 1. Pure Functional Graph Operators (Graph Immutability)
Graph modification functions (e.g. lesions, subnetwork slicing) must **never** mutate the input graph in-place. Always operate on explicit deep copies:
```python
# CORRECT
def apply_lesion(graph: nx.DiGraph, nodes_to_remove: list) -> nx.DiGraph:
    lesioned_graph = graph.copy()
    lesioned_graph.remove_nodes_from(nodes_to_remove)
    return lesioned_graph

# INCORRECT (Forbidden)
def apply_lesion(graph: nx.DiGraph, nodes_to_remove: list):
    graph.remove_nodes_from(nodes_to_remove)  # Mutates caller graph!
```

### 2. Strict Determinism & Seed Transparency
Never invoke unseeded random generators. Always accept and log explicit pseudo-random seeds:
```python
# CORRECT
rng = np.random.default_rng(seed)
selected_nodes = rng.choice(candidates, size=k, replace=False)

# INCORRECT (Forbidden)
import random
selected_nodes = random.sample(candidates, k)  # Unseeded!
```

### 3. Zero-Invention Data Policy
- Never interpolate, synthesize, or fabricate neuron IDs, synaptic contacts, or cell-type annotations.
- Every anatomical edge weight must reflect authentic electron-microscopy reconstructed synapse counts from authoritative repositories.

### 4. Memory-Safe Parquet Streaming
- Large tabular conversions must use chunked PyArrow streaming with peak memory capped below $50\,\text{MB}$.
- Never load multi-gigabyte raw tables entirely into memory via unconstrained `pandas.read_csv()`.

### 5. Numerical Edge-Case Safety
- Handle division by zero safely (e.g., when intact firing rate is $0\,\text{Hz}$, activity robustness must return `NaN` or a well-defined fallback, never infinity or unhandled exceptions).
- Ensure integration timesteps and coupling parameters do not trigger silent numeric overflows or `NaN` membrane potentials.

---

## 📝 Code Style & Guidelines

- **PEP 8 Compliance**: Follow standard Python conventions.
- **Type Annotations**: Add type hints to all public functions in `src/`.
- **Docstrings**: Document parameters, return values, units (e.g., $\text{mV}$, $\text{ms}$, $\text{Hz}$), and biophysical/computational interpretation.
- **Cautionary Language**: Follow the four-level interpretation policy described in [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md). Distinguish measured computational model results from live biological claims.

---

## 🧪 Testing Guidelines

- Any new functionality added to `src/` must be accompanied by comprehensive unit tests in `tests/`.
- Tests must be fast, deterministic, and self-contained (avoiding external network downloads during testing).
- Run full test coverage:
  ```bash
  pytest -v --durations=10
  ```

---

## 🔀 Git Workflow & Commit Guidelines

We use **Conventional Commits**:
- `feat:` A new feature or experiment script (e.g., `feat: implement Phase 8 PageRank lesion strategy`).
- `fix:` A bug fix in simulation or analysis code (e.g., `fix: correct zero-division check in stimulus robustness`).
- `docs:` Documentation improvements or roadmaps (e.g., `docs: update Phase 8 statistical plan`).
- `test:` Adding or refining unit tests (e.g., `test: add invariants test for stratified null model`).
- `perf:` Performance optimizations without behavioral changes (e.g., `perf: vectorize membrane reset check`).

### Submitting a Pull Request (PR)

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feat/phase8-topology-characterization
   ```
2. Commit your changes following conventional commit syntax.
3. Verify that all 64 existing tests pass:
   ```bash
   pytest -v
   ```
4. Verify independent reproducibility:
   ```bash
   python scratch/verify_phase7_reproducibility.py
   ```
5. Push to your fork and submit a Pull Request against `main`.

---

## 📋 Pull Request Checklist

Before submitting a PR, ensure:
- [ ] All 64 unit tests pass (`pytest -v`).
- [ ] New features include corresponding unit tests in `tests/`.
- [ ] No locked Phase 1–7 CSVs or figures in `results/` were modified.
- [ ] Code adheres to graph immutability and deterministic seed invariants.
- [ ] Documentation (`README.md`, `docs/`) is updated where appropriate.
- [ ] No temporary files, logs, caches, or virtual environments are tracked.

Thank you for contributing to open, reproducible connectome neurodynamics!
