# Virtual Lesion Analysis of a Drosophila Connectome Subnetwork (Phase 4)

## 1. Research Question & Objective

How does targeted structural perturbation (ablation of high-degree hub neurons) compare to uniform random neuron ablation in degrading structural connectivity and simulated spiking dynamics in a connectome-constrained neural circuit?

### Neutral Hypothesis
We hypothesize that in a network with heavy-tailed degree distributions, targeted removal of highest-degree nodes will cause a faster decline in structural edge count and total synaptic weight than uniform random ablation. However, whether simulated population firing rates degrade more rapidly under hub removal than under random removal remains an empirical question, as dynamical activity in a recurrent Leaky Integrate-and-Fire (LIF) network depends nonlinearly on both recurrent excitation and external drive distribution.

---

## 2. Connectome Dataset & Subnetwork Extraction

### 2.1 Empirical Dataset
- **Source:** Janelia FlyEM Drosophila Hemibrain Connectome v1.2.1 (Scheffer et al., 2020).
- **Full Connectome Size:** 21,739 proofread neurons, 3,550,403 unique directed neuron pairs, 14,329,229 total reconstructed synaptic contacts (PSDs).
- **Data Ingestion:** Extracted from raw archive into schema-validated Apache Parquet tables (`neurons.parquet`, `connections.parquet`). Multi-ROI parallel connections between identical pre- and post-synaptic neuron pairs were aggregated by summing synaptic contact counts; self-loops were excluded.

### 2.2 1,000-Neuron Subnetwork Selection
To run computationally demanding spiking simulations locally without high-end GPU hardware while preserving structural connectivity density:
- **Selection Strategy:** Highest total degree ($k_{\text{total}} = k_{\text{in}} + k_{\text{out}}$).
- **Subnetwork Size:** $N = 1,000$ neurons.
- **Subnetwork Density & Edges:** 113,089 directed edges (density $\rho = 0.1132$).
- **Total Synaptic Weight:** 937,272 PSD contacts ($\approx 8.29$ synapses per connected pair).
- **Component Structure:** Single weakly connected component ($100\%$ membership).

---

## 3. Experimental Design & Lesion Protocol

All lesions were applied directly to the graph topology **prior** to running numerical integration. The original baseline subnetwork graph was never mutated; each lesion was executed on an isolated copy of the network.

### 3.1 Experimental Conditions

| Condition | Strategy | Fractions Removed | Removal Count ($N=1000$) | Seeds / Repetitions | Tie Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Baseline** | None | 0% | 0 | Seed 42 (1 run) | N/A |
| **B. Random Lesion** | Uniform random selection without replacement | 1%, 5%, 10%, 20% | 10, 50, 100, 200 | 5 seeds: 42, 123, 456, 789, 1000 (20 runs) | N/A |
| **C. Targeted Hub Lesion** | Highest pre-lesion total degree ($k_{\text{in}} + k_{\text{out}}$) | 1%, 5%, 10%, 20% | 10, 50, 100, 200 | Deterministic (4 runs) | Ascending numerical node ID (`(-degree, node_id)`) |

**Total Experimental Runs:** 25 simulations (1 Baseline + 20 Random + 4 Hub).

### 3.2 Invariants Maintained During Lesioning
1. **Reproducibility:** Seeded pseudo-random generator (`np.random.default_rng(seed)`) ensures bitwise identical node selection across independent executions.
2. **Deterministic Tie-Breaking:** For hub lesioning, nodes with equal degree are ranked deterministically by node ID: `key=lambda x: (-x[1], x[0])`.
3. **Graph Immutability:** Pre-lesion graphs are never modified in-place; all deletions operate on explicit copies.
4. **Subgraph Invariant:** Resulting edge count $\le$ baseline edge count; every remaining edge is a strict subset of the original edge set.

---

## 4. Spiking Model Parameters & Coupling Scale Interpretation

### 4.1 Fixed Leaky Integrate-and-Fire (LIF) Parameters
The baseline LIF parameters were held strictly constant across all 25 simulation runs:

| Parameter | Symbol | Value | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| Membrane time constant | $\tau_m$ | 20.0 | ms | Passive decay constant |
| Resting potential | $V_{\text{rest}}$ | -65.0 | mV | Leak reversal potential |
| Reset potential | $V_{\text{reset}}$ | -70.0 | mV | Hyperpolarization after spike |
| Spike threshold | $V_{\text{threshold}}$ | -50.0 | mV | Threshold for action potential emission |
| Refractory period | $t_{\text{ref}}$ | 2.0 | ms | Absolute refractory duration |
| Integration timestep | $dt$ | 0.5 | ms | Forward Euler time resolution |
| Simulation duration | $T$ | 1000.0 | ms | Total simulation epoch |
| Injected current | $I_{\text{ext}}$ | 18.0 | mV/ms eq. | Excitatory stimulus amplitude |
| Stimulus window | $[t_{\text{on}}, t_{\text{off}}]$ | [200, 600] | ms | Transient stimulation pulse |
| Stimulated fraction | $f_{\text{stim}}$ | 0.25 | — | Fraction of population directly stimulated |
| Synaptic weight scale | $\alpha$ | 0.01 | mV/synapse | Computational coupling scale |

### 4.2 Explicit Scientific Interpretation of $\alpha = 0.01$
In the dense 1,000-neuron subnetwork, each neuron receives on average $\approx 113$ presynaptic partner connections and $\approx 937$ total incoming anatomical synapses (PSDs). If anatomical synapse counts were mapped directly to unscaled post-synaptic potentials ($\alpha \ge 0.05\,\text{mV/synapse}$), recurrent excitation would trigger an immediate avalanche of runaway excitation ($V > 120\,\text{mV}$ or firing rates exceeding $350\,\text{Hz}$).

Therefore:
$$\alpha = 0.01\,\text{mV/synapse}$$
**is a computationally stable coupling scale, NOT an empirically calibrated biological synaptic efficacy.** It serves to place the dense subnetwork into a non-exploding, sparse-firing dynamical regime ($2.28\,\text{Hz}$ baseline) suitable for studying comparative network perturbation.

---

## 5. Measured Research Metrics

For each condition, structural and functional metrics were logged:

### 5.1 Structural Metrics
1. **Remaining Neurons ($N_{\text{rem}}$) & Edges ($E_{\text{rem}}$)**
2. **Network Robustness:** $R_{\text{net}} = E_{\text{rem}} / E_{\text{baseline}}$
3. **Remaining Synaptic Weight Fraction:** $W_{\text{rem}} / W_{\text{baseline}}$
4. **Giant Weakly Connected Component (WCC) Fraction:** $\text{WCC}_{\text{rem}} / \text{WCC}_{\text{baseline}}$

### 5.2 Dynamical Activity Metrics
1. **Total Population Spikes:** Emitted across 1000 ms.
2. **Mean Firing Rate ($\bar{\nu}$):** $\frac{1}{N_{\text{rem}} \cdot T} \sum_i \text{spikes}_i$ (Hz).
3. **Median & Max Firing Rate (Hz)**
4. **Active Neuron Fraction:** Percentage of surviving neurons emitting $\ge 1$ spike.
5. **Activity Robustness:** $R_{\text{act}} = \bar{\nu}_{\text{lesion}} / \bar{\nu}_{\text{baseline}}$
6. **Relative Change in Firing Rate:** $(\bar{\nu}_{\text{lesion}} - \bar{\nu}_{\text{baseline}}) / \bar{\nu}_{\text{baseline}}$

---

## 6. Scientific Limitations & Selection Bias

### 6.1 Subnetwork Selection Bias
- **Core-Periphery Bias:** Selecting the top 1,000 neurons by total degree enriches heavily for dense central neuropil hubs (central complex, mushroom body output/input hubs, antennal lobe projection hubs) rather than a representative cross-section of the whole fly brain.
- **Over-Recurrence:** Average in-degree in this subnetwork is $\approx 113$, compared to $\approx 163$ across the entire 21.7K Hemibrain. However, within this 1,000-neuron subnetwork, the graph density ($\rho = 0.1132$) is substantially higher than whole-brain density ($\rho \approx 0.0075$).
- **Edge Truncation:** Connections to and from the remaining 20,739 neurons in the Hemibrain were severed by subnetwork extraction.

### 6.2 Computational vs. Biological Brain Resilience
- **Ablation vs. Real Damage:** In a biological organism, neuronal injury triggers microglial/glial responses, excitotoxicity, compensatory homeostatic plasticity, altered neuromodulatory tone (dopamine, serotonin, octopamine), and behavioral feedback loops. This virtual lesion model captures only instantaneous topological deletion in an idealized feedforward/recurrent LIF network.
- **No Behavioral Equivalence:** A drop in simulated firing rate in this subnetwork does NOT imply fly paralysis, memory impairment, or death.
- **Descriptive Statistics:** In the absence of hypothesis-driven inferential tests with pre-registered effect sizes, all comparisons are treated strictly as descriptive computational benchmarks.

---

## 7. Output Artifacts & Verification

- **Raw Data Table:** `results/tables/lesion_raw_results.csv` (25 individual simulation trajectories)
- **Summary Table:** `results/tables/lesion_summary.csv` (aggregated mean $\pm$ standard deviation, min, max)
- **Topological Metrics Table:** `results/tables/lesion_graph_metrics.csv`
- **Publication Figures:**
  - `results/figures/lesion_firing_rate_comparison.png`
  - `results/figures/lesion_spike_count_comparison.png`
  - `results/figures/lesion_active_fraction.png`
  - `results/figures/lesion_network_connectivity.png`
  - `results/figures/lesion_effect_summary.png`
