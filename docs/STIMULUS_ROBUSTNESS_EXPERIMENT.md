# Stimulus-Dependence of Lesion Robustness (Phase 5)

## 1. Research Question & Rationale

In Phase 4, virtual lesion experiments on the real 1,000-neuron Janelia FlyEM Hemibrain connectome subnetwork revealed an apparent paradox: removing 20% of high-degree hub neurons eliminated **58.2%** of all synaptic edges, yet population mean firing rate remained buffered ($2.25\,\text{Hz}$ vs $2.28\,\text{Hz}$ baseline; activity robustness $R_{\text{act}} = 0.987$).

However, the Phase 4 protocol applied a strong external current pulse to **25% of the network** ($I_{\text{ext}} = 18.0\,\text{mV}$). This raised a fundamental methodological question:

> **Does the apparent robustness of network activity to neuronal lesions persist when external stimulation is progressively reduced, or was the Phase 4 resilience partly masked by strong external drive?**

Phase 5 was designed to systematically decouple the role of external drive from intrinsic recurrent network dynamics by varying the fraction of directly stimulated neurons across five orders of drive.

---

## 2. Testable Hypothesis

> *If the apparent activity robustness observed in Phase 4 was an artifact of widespread direct external drive overriding recurrent dynamics, then progressively reducing the stimulated fraction will unmask greater lesion vulnerability—evidenced by a steeper collapse in activity robustness, lower active neuron fractions, and divergent post-stimulus dynamics.*

This hypothesis is treated neutrally as a computational inquiry, not a foregone conclusion.

---

## 3. Experimental Design & Methodology

### 3.1 Network Topology & LIF Model (Frozen from Phase 4)
- **Dataset:** Janelia FlyEM Drosophila Hemibrain v1.2.1.
- **Subnetwork:** 1,000 neurons, 113,089 directed edges, 937,685 anatomical synapses ($k_{\text{total}}$ selection, seed 42).
- **LIF Parameters:**
  - $\tau_m = 20.0\,\text{ms}$, $V_{\text{rest}} = -65.0\,\text{mV}$, $V_{\text{reset}} = -70.0\,\text{mV}$, $V_{\text{threshold}} = -50.0\,\text{mV}$
  - Refractory period: $t_{\text{ref}} = 2.0\,\text{ms}$, Euler step: $dt = 0.5\,\text{ms}$, Duration: $T = 1,000.0\,\text{ms}$
  - Synaptic coupling scale: $\alpha = 0.01\,\text{mV/synapse}$ (computational coupling scale, not biologically calibrated quantal efficacy)
  - External pulse current: $I_{\text{ext}} = 18.0\,\text{mV}$, window = $200–600\,\text{ms}$

### 3.2 Stimulation Protocol & Lesion Coupling
- **Five Stimulus Fractions:**
  - $25\%$ ($250$ neurons) — Phase 4 baseline condition
  - $10\%$ ($100$ neurons) — reduced external drive
  - $5\%$ ($50$ neurons) — low external drive
  - $1\%$ ($10$ neurons) — very weak external drive
  - $0\%$ ($0$ neurons) — zero external drive (spontaneous activity test)
- **Selection Rule:** Stimulated neurons were selected deterministically on the **intact 1,000-neuron network before lesioning**.
- **Survival Semantics:** When lesions were applied, any lesioned stimulated neurons were eliminated naturally. Surviving stimulated neurons continued receiving stimulation; no re-sampling or artificial replacement was performed.

### 3.3 Lesion Protocol & Experimental Matrix
- **Lesion Levels:** $0\%$ (intact baseline), $5\%$ ($50$ neurons ablated), $10\%$ ($100$ neurons ablated), $20\%$ ($200$ neurons ablated).
- **Lesion Strategies:**
  - **Intact:** 5 stimulus conditions $\times 1 = 5$ simulations.
  - **Targeted Hub Lesion:** Pre-lesion total degree descending with tie-break `(-degree, node_id)`: 5 stimulus conditions $\times 3$ lesion fractions $= 15$ simulations.
  - **Random Lesion:** 5 stimulus conditions $\times 3$ lesion fractions $\times 5$ seeds (`42`, `123`, `456`, `789`, `1000`) $= 75$ simulations.
- **Total Simulation Matrix:** **95 full simulations**.

### 3.4 Stimulus-Specific Baseline Normalization
Every stimulus condition was normalized against its **own intact (0% lesion) baseline**:
$$R_{\text{act}} = \frac{\bar{\nu}_{\text{lesion}}(f_{\text{stim}})}{\bar{\nu}_{\text{intact}}(f_{\text{stim}})}$$
When baseline activity is zero ($0\%$ stimulation), robustness is formally undefined ($0/0$) and recorded as `NaN` to avoid generating invalid infinities.

### 3.5 Temporal Decomposition
- **Stimulus Window ($200–600\,\text{ms}$):** Direct pulse current active ($T_{\text{stim}} = 400\,\text{ms}$).
- **Post-Stimulus Window ($600–1,000\,\text{ms}$):** External drive removed ($T_{\text{post}} = 400\,\text{ms}$) to measure self-sustained reverberant activity.
- **Stimulus-to-Post Ratio:** $\bar{\nu}_{\text{stim}} / \bar{\nu}_{\text{post}}$ (evaluated safely, yielding `NaN` when post-stimulus activity is zero).

---

## 4. Measured Experimental Results

### 4.1 Intact Baselines Across Stimulation Levels

| Stimulus Fraction | Stimulated Count | Intact Mean Rate (Hz) | Total Spikes | Active Fraction | Stim Window Rate (Hz) | Post-Stim Window Rate (Hz) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **25%** | 250 | $2.28\,\text{Hz}$ | 2,284 | $25.6\%$ | $5.71\,\text{Hz}$ | $0.00\,\text{Hz}$ |
| **10%** | 100 | $0.90\,\text{Hz}$ | 900 | $10.0\%$ | $2.25\,\text{Hz}$ | $0.00\,\text{Hz}$ |
| **5%** | 50 | $0.45\,\text{Hz}$ | 450 | $5.0\%$ | $1.12\,\text{Hz}$ | $0.00\,\text{Hz}$ |
| **1%** | 10 | $0.09\,\text{Hz}$ | 90 | $1.0\%$ | $0.23\,\text{Hz}$ | $0.00\,\text{Hz}$ |
| **0%** | 0 | $0.00\,\text{Hz}$ | 0 | $0.0\%$ | $0.00\,\text{Hz}$ | $0.00\,\text{Hz}$ |

*Key Observation:* In all intact baselines, post-stimulus firing rate was strictly $0.00\,\text{Hz}$. When the external pulse turns off at $t = 600\,\text{ms}$, recurrent activity immediately ceases. Furthermore, active neuron count matches the stimulated neuron count almost 1:1, indicating that with $\alpha = 0.01$, recurrent excitation is predominantly subthreshold and does not drive un-stimulated neurons above threshold.

### 4.2 Activity Robustness Under 20% Neuronal Ablation

| Stimulus Fraction | Hub Firing Rate (20% Lesion) | Hub Activity Robustness | Random Firing Rate (20% Lesion) | Random Activity Robustness | Surviving Stimulated (Hub) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **25%** | $2.31\,\text{Hz}$ | **$1.013$** | $2.29 \pm 0.05\,\text{Hz}$ | **$1.006 \pm 0.022$** | 205 / 250 ($82.0\%$) |
| **10%** | $0.88\,\text{Hz}$ | **$0.978$** | $0.91 \pm 0.02\,\text{Hz}$ | **$1.016 \pm 0.026$** | 78 / 100 ($78.0\%$) |
| **5%** | $0.41\,\text{Hz}$ | **$0.911$** | $0.43 \pm 0.02\,\text{Hz}$ | **$0.964 \pm 0.034$** | 36 / 50 ($72.0\%$) |
| **1%** | $0.09\,\text{Hz}$ | **$1.000$** | $0.09 \pm 0.01\,\text{Hz}$ | **$1.044 \pm 0.099$** | 8 / 10 ($80.0\%$) |
| **0%** | $0.00\,\text{Hz}$ | **NaN** | $0.00 \pm 0.00\,\text{Hz}$ | **NaN** | 0 / 0 |

### 4.3 Structural Degradation Invariants
- Pre-lesion network: 1,000 neurons, 113,089 edges.
- 5% Hub Lesion: 86,381 edges ($76.4\%$ edge retention, $76.5\%$ weight retention).
- 10% Hub Lesion: 69,570 edges ($61.5\%$ edge retention, $60.2\%$ weight retention).
- 20% Hub Lesion: 47,301 edges ($41.8\%$ edge retention, $40.7\%$ weight retention).
- 20% Random Lesion (Mean across 5 seeds): $73,119 \pm 625$ edges ($64.7\%$ edge retention).

---

## 5. Scientific Interpretation & Core Findings

1. **Under the tested parameters, direct external stimulation dominated observed spiking activity:**
   Under this specific LIF parameterization ($\alpha = 0.01\,\text{mV / synapse contact}$, $I_{\text{ext}} = 18.0\,\text{mV}$), network spiking was almost entirely feedforward-dominated by the directly stimulated neurons. Each surviving stimulated neuron fired regular spikes ($\sim 22.5\,\text{Hz}$ in the 400 ms stimulus window; exactly 9 spikes per neuron), behaving nearly identically to uncoupled, isolated LIF neurons receiving $I_{\text{ext}} = 18.0\,\text{mV}$. Unstimulated neurons received subthreshold EPSPs that were insufficient to cross threshold ($V_{\text{threshold}} = -50.0\,\text{mV}$).

2. **Phase 4 Apparent "Robustness" Explained by Proportional Normalization:**
   The apparent stability of the population mean firing rate under lesions observed in Phase 4 ($\bar{\nu} \approx 2.25–2.28\,\text{Hz}$) does **not** demonstrate functional network resilience or active biological compensation. Rather, it arises as an arithmetic consequence of proportional reduction:
   $$\bar{\nu} = \frac{\text{Total Spikes}}{N_{\text{surviving}} \cdot T}$$
   **Lesion removes neurons $\rightarrow$ directly stimulated neurons also decrease $\rightarrow$ total spikes decrease $\rightarrow$ surviving-neuron normalization ($N_{\text{surviving}}$) decreases at roughly the same rate.**
   For example, in a 20% hub lesion, surviving stimulated neurons drop from 250 to 205 ($18\%$ reduction), and total spikes drop from 2,284 to 1,845 ($19\%$ reduction). Because the population metric normalizes total spikes by the surviving neuron count ($N_{\text{surviving}} = 800$), both numerator and denominator shrink at roughly the same rate. This explains why the population rate remained relatively constant despite massive structural disconnection ($58.2\%$ edge loss).

3. **Absence of Post-Stimulus Activity Under Tested Coupling Scale (No Critical Threshold Claim):**
   Post-stimulus firing rate was strictly $0.00\,\text{Hz}$ across all 95 simulation runs.
   > **“Under the tested coupling scale ($\alpha = 0.01\,\text{mV / synapse contact}$), recurrent excitation was insufficient to sustain activity after external stimulation ceased.”**
   
   *Scientific Precaution:* This experiment tested a single fixed coupling scale ($\alpha = 0.01$) rather than a systematic fine-grained parametric sweep around a bifurcation point. Therefore, **we do not claim to have identified a critical reverberation threshold**. We only establish that at this specific coupling scale, recurrent activity does not persist after stimulus offset.

4. **Stimulus-Dependent Lesion Sensitivity:**
   When external drive is reduced (e.g., from 25% down to 5%), lesion-dependent divergence becomes more apparent:
   - At 25% stimulation: Hub 20% robustness is $1.013$, Random 20% is $1.006 \pm 0.022$.
   - At 5% stimulation: Hub 20% robustness drops to $0.911$, Random 20% drops to $0.964 \pm 0.034$.
   Thus, reducing external drive exposes stronger lesion-induced effects, though the magnitude of the drop remains relatively modest in this weakly coupled regime.

5. **Behavior at Zero External Drive (0% Stimulus Condition):**
   In the complete absence of external stimulation ($0\%$), deterministic LIF initialization at resting potential produces zero action potentials across all intact and lesioned conditions ($0\,\text{Hz}$). Spontaneous activity is absent without external current or intrinsic pacemaker mechanisms, and activity robustness is mathematically undefined ($0/0$), recorded as `NaN`.

---

## 6. Methodological Limitations

1. **Weakly Coupled Regime vs. Strongly Interacting Network:** Under the tested parameter regime ($\alpha = 0.01\,\text{mV / synapse contact}$, $I_{\text{ext}} = 18.0\,\text{mV}$), the current network configuration behaves more like an ensemble of independently driven LIF neurons connected by relatively weak recurrent coupling than a strongly interacting recurrent network.
2. **Coupling Scale Units ($\alpha$):** The synaptic coupling scale $\alpha = 0.01\,\text{mV / synapse contact}$ represents the postsynaptic potential jump per reconstructed anatomical synapse. When a presynaptic neuron spikes, the postsynaptic membrane potential instantaneously jumps by $\Delta V_i = \alpha \times W_{ji}\,\text{mV}$, where $W_{ji}$ is the integer synapse count (PSD count).
3. **Partial Connectome & Hub Enrichment:** The 1,000-neuron subnetwork is enriched for highest-degree neurons from the central brain (hemibrain), lacking peripheral sensory and motor structures.
4. **Absence of Inhibitory Cell Types:** Neurotransmitter identities (GABAergic, glutamatergic, cholinergic) were not differentiated; all connections were modeled as excitatory integrate-and-fire inputs.
5. **Deterministic Resting Potential:** Real Drosophila neurons exhibit spontaneous fluctuations, miniature EPSPs, and neuromodulatory tone (dopaminergic/octopaminergic modulation) not present in this minimal deterministic model.
6. **No Biological Resilience Claim:** Stability of firing rates under virtual lesions reflects computational properties and normalization arithmetic of an integrate-and-fire model under external drive, not biological resilience or behavioral preservation in living flies.

---

## 7. Generated Tables and Figures

### Output Tables
- `results/tables/stimulus_robustness_raw_results.csv` (all 95 simulations)
- `results/tables/stimulus_robustness_summary.csv` (aggregated statistics across stimulus and lesion levels)
- `results/tables/stimulus_robustness_graph_metrics.csv` (topological and stimulus coverage metrics)

### Output Figures
- `results/figures/stimulus_mean_firing_rate.png` (Figure 1: Mean rate vs lesion % across all stimulus levels)
- `results/figures/stimulus_activity_robustness.png` (Figure 2: Normalized activity robustness vs lesion %)
- `results/figures/stimulus_spike_count.png` (Figure 3: Total spike counts vs lesion %)
- `results/figures/stimulus_active_fraction.png` (Figure 4: Active neuron fraction vs lesion %)
- `results/figures/stimulus_window_vs_post_rate.png` (Figure 5: Stimulus window vs post-stimulus firing rates)
- `results/figures/stimulus_edge_retention_vs_robustness.png` (Figure 6: Structural edge retention vs functional activity robustness)
- `results/figures/stimulus_effective_fraction.png` (Figure 7: Surviving directly stimulated neurons after lesions)
- `results/figures/stimulus_robustness_heatmap.png` (Figure 8: 2D heatmaps of activity robustness)
