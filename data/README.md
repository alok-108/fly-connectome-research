# Connectome Data Architecture & Ingestion Guide

This directory enforces a strict pipeline architecture:

$$\text{RAW ARCHIVE} \longrightarrow \text{STREAMING CONVERSION} \longrightarrow \text{PARQUET TABLES} \longrightarrow \text{EXPERIMENTS}$$

---

## 1. Raw Data (`data/raw/`)
Raw, immutable datasets downloaded from official, open-access public repositories.
**Files inside `data/raw/` are excluded from Git version control by `.gitignore`.**

### Primary Dataset: Janelia FlyEM Drosophila Hemibrain (v1.2.1)
- **Biological Scope**: Central brain of an adult female *Drosophila melanogaster* (~21,739 proofread neurons, ~4.26M synapse entries).
- **Primary Source**: HHMI Janelia Research Campus & Google Research (Scheffer et al., 2020, *eLife*).
- **Public Mirror**: Netzschleuder Network Catalogue (curated by Dr. Tiago Peixoto): [https://networks.skewed.de/net/fly_hemibrain](https://networks.skewed.de/net/fly_hemibrain)
- **Direct Archive**: `fly_hemibrain.csv.zip` (~16.1 MiB compressed; contains `nodes.csv` [21,739 rows] and `edges.csv` [4,259,624 rows]).
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

### How to Download the Raw Dataset:
If reproducing from a clean repository clone:
```powershell
# In PowerShell:
Invoke-WebRequest -Uri "https://networks.skewed.de/net/fly_hemibrain/files/fly_hemibrain.csv.zip" -OutFile "data/raw/fly_hemibrain.csv.zip"
```
Or via `curl`:
```bash
curl -L -o data/raw/fly_hemibrain.csv.zip "https://networks.skewed.de/net/fly_hemibrain/files/fly_hemibrain.csv.zip"
```

---

## 2. Processed Data (`data/processed/`)
Standardized, indexed, and sanitized tabular representations optimized for fast subnetwork extraction and local memory constraints:
- `neurons.parquet`: Standardized node table (21,739 proofread neurons, ~0.4 MB)
  - `neuron_id` (int64): Unique 0-indexed identifier
  - `body_id` (int64): Janelia FlyEM body ID
  - `cell_type` (str): Morphological cell type classification
  - `instance` (str): Unique instance identifier
  - `neuropil` (str): Primary neuropil compartment
- `connections.parquet`: Standardized directed edge table (4,259,624 directed edges, ~14.5 MB)
  - `source` (int64): Presynaptic neuron ID
  - `target` (int64): Postsynaptic neuron ID
  - `weight` (int64): Reconstructed synaptic contact count (PSDs)
  - `roi` (str): Neuropil compartment of synaptic contact

### How to Rebuild Processed Parquet Data:
To convert the raw zip archive into processed Parquet files with peak memory $< 30\,\text{MB}$:
```powershell
python -c "from src.data_loader import convert_raw_to_parquet; convert_raw_to_parquet()"
```

---

## 3. Provenance & Scientific Lineage
For full dataset comparison, licensing, and schema verification, see [`data/DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
