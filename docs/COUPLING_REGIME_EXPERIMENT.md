# Coupling-Regime Sweep and Recurrent Dynamics (Phase 6)

## 1. Research Question & Rationale

In Phase 5, virtual lesion experiments on the real 1,000-neuron Janelia FlyEM Hemibrain connectome subnetwork demonstrated that under the baseline coupling scale ($\alpha = 0.01\,\text{mV / synapse contact}$) and 25% direct external drive ($I_{\text{ext}} = 18.0\,\text{mV}$):
- Observed network spiking was dominated by the directly stimulated neurons.
- Unstimulated neurons remained largely subthreshold.
- Post-stimulus firing ceased immediately ($0.00\,\text{Hz}$ across all 95 simulations).
- Apparent firing rate stability under lesions was driven by the arithmetic of proportional reduction and surviving-neuron normalization ($N_{\text{surviving}}$).

This raised the central scientific question for Phase 6:

> **Does the weak influence of recurrent connectome coupling observed at $\alpha = 0.01$ persist across coupling strengths, or does the network enter a regime where recurrent connectivity materially affects activity?**

Phase 6 was designed to systematically answer this question by sweeping the synaptic coupling scale $\alpha$ across 11 values ($0.000$ to $0.050\,\text{mV / synapse contact}$) across five external drive conditions (25%, 10%, 5%, 1%, 0%), followed by limited lesion validation at measured representative coupling regimes.

---

## 2. Experimental Design & Methodology

### 2.1 Dataset & LIF Model Parameters (Frozen from Phases 3–5)
- **Connectome Dataset:** Janelia FlyEM Drosophila Hemibrain v1.2.1.
- **Subnetwork:** 1,000 neurons, 113,089 directed edges, 937,685 anatomical synapses ($k_{\text{total}}$ selection, seed 42).
- **LIF Parameters (Frozen):**
  - $\tau_m = 20.0\,\text{ms}$, $V_{\text{rest}} = -65.0\,\text{mV}$, $V_{\text{reset}} = -70.0\,\text{mV}$, $V_{\text{threshold}} = -50.0\,\text{mV}$
  - Absolute refractory period: $t_{\text{ref}} = 2.0\,\text{ms}$, Euler step: $dt = 0.5\,\text{ms}$, Duration: $T = 1,000.0\,\text{ms}$
  - External current pulse: $I_{\text{ext}} = 18.0\,\text{mV}$, window = $200.0–600.0\,\text{ms}$ ($T_{\text{stim}} = 400\,\text{ms}$)
  - Background membrane noise: $\sigma_{\text{noise}} = 1.0$
- **Sole Primary Varied Parameter:** Synaptic coupling scale $\alpha \in [0.000, 0.050]\,\text{mV / synapse contact}$.
  *Note:* Units are $\text{mV / synapse contact}$, representing the instantaneous postsynaptic membrane potential jump per anatomical synaptic contact ($W_{ji}$) upon presynaptic spike arrival: $\Delta V_i = \alpha \cdot W_{ji}\,\text{mV}$.

### 2.2 Coupling Sweep Grid (11 Values)
$$\alpha \in \{0.000, 0.001, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050\}\,\text{mV / synapse contact}$$

### 2.3 Stimulus Conditions
- **Primary:** $25\%$ ($250$ neurons) — fixed deterministic set selected on intact network (seed 42).
- **Secondary:** $10\%$ ($100$ neurons), $5\%$ ($50$ neurons), $1\%$ ($10$ neurons), $0\%$ ($0$ neurons).
- Total intact simulations: $11\,\alpha \times 5\,\text{stimulus levels} = \mathbf{55}$ simulations.

### 2.4 Activity Windows
- **Pre-stimulus Window ($0–200\,\text{ms}$):** Baseline spontaneous / resting dynamics before pulse onset.
- **Stimulus Window ($200–600\,\text{ms}$):** External drive active ($I_{\text{ext}} = 18.0\,\text{mV}$).
- **Post-stimulus Window ($600–1,000\,\text{ms}$):** External drive removed ($I_{\text{ext}} = 0\,\text{mV}$) to measure self-sustained recurrent persistence.
- **Full Simulation ($0–1,000\,\text{ms}$):** Global metrics across the entire 1-second simulation.

### 2.5 Recurrent & Connectivity Metrics
- **Unstimulated Recruitment:** Recruited neuron count and fraction among unstimulated neurons ($N = 750$) producing $\ge 1$ spike during the stimulus window.
- **Stimulated vs. Unstimulated Rates:** Separate firing rate tracking for stimulated vs. unstimulated neurons.
- **Post-stimulus Persistence:** Ratio of post-stimulus active fraction to stimulus-window active fraction ($\frac{f_{\text{active, post}}}{f_{\text{active, stim}}}$).
- **Recurrent Synaptic Input Vector:** $\mathbf{I}_{\text{syn}} = \alpha \cdot (W^T \mathbf{s})$, partitioned by presynaptic source (stimulated vs. unstimulated).

### 2.6 Numerical Stability & Regime Characterization
- Instability detected if: (1) $\text{NaN}/\text{Inf}$ in $V_m$, (2) $V_m > 120.0\,\text{mV}$ (membrane explosion), or (3) max firing rate $> 350.0\,\text{Hz}$.
- Descriptive observed regime characterization:
  - **Low-Coupling Regime (approximately $\alpha = 0.000–0.010$):** Little or no unstimulated recruitment, no persistent post-stimulus activity, activity dominated by direct stimulation.
  - **Transition / Recurrent-Recruitment Regime (approximately $\alpha = 0.015–0.030$):** Increasing unstimulated recruitment, increasing population activity, post-stimulus activity emerges at $\alpha = 0.020$, strong recurrent recruitment at $\alpha = 0.030$.
  - **Numerically Unstable Regime ($\alpha \ge 0.040$ within the tested grid):** Numerical instability detected under the excitatory-only model.

---

## 3. Measured Intact Coupling Results (Primary 25% Drive)

| $\alpha$ (mV/synapse) | Stable? | Observed Dynamic Regime | Mean Rate (Hz) | Active Fraction | Recruited Unstimulated | Post-Stim Rate (Hz) | $V_{\max}$ (mV) | Total Spikes |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0000** | Yes | Externally Driven (Weak Coupling) | $2.25$ | $25.0\%$ | $0$ ($0.0\%$) | $0.000$ | $-50.0$ | 2,250 |
| **0.0010** | Yes | Externally Driven (Weak Coupling) | $2.25$ | $25.0\%$ | $0$ ($0.0\%$) | $0.000$ | $-48.1$ | 2,250 |
| **0.0025** | Yes | Externally Driven (Weak Coupling) | $2.25$ | $25.0\%$ | $0$ ($0.0\%$) | $0.000$ | $-45.2$ | 2,250 |
| **0.0050** | Yes | Externally Driven (Weak Coupling) | $2.25$ | $25.0\%$ | $0$ ($0.0\%$) | $0.000$ | $-39.4$ | 2,250 |
| **0.0075** | Yes | Externally Driven (Weak Coupling) | $2.26$ | $25.1\%$ | $1$ ($0.13\%$) | $0.000$ | $-34.1$ | 2,259 |
| **0.0100** | Yes | Externally Driven (Weak Coupling) | $2.28$ | $25.6\%$ | $6$ ($0.80\%$) | $0.000$ | $-28.8$ | 2,283 |
| **0.0150** | Yes | Recurrent Recruitment | $2.68$ | $27.8\%$ | $28$ ($3.73\%$) | $0.000$ | $-18.4$ | 2,683 |
| **0.0200** | Yes | Sustained Recurrent Activity | $4.41$ | $39.6\%$ | $146$ ($19.47\%$) | **$0.060$** | $-19.1$ | 4,409 |
| **0.0300** | Yes | Sustained Recurrent Activity | **$19.48$** | **$90.3\%$** | **$653$ ($87.07\%$)** | **$1.558$** | **$+9.8$** | **19,477** |
| **0.0400** | **No** | Unstable (Explosion at $t=267.5\,\text{ms}$) | $0.88$ | $70.5\%$ | $455$ ($60.67\%$) | $0.000$ | $+123.3$ | 884 |
| **0.0500** | **No** | Unstable (Explosion at $t=234.0\,\text{ms}$) | $0.34$ | $34.1\%$ | $92$ ($12.27\%$) | $0.000$ | $+123.2$ | 341 |

*Observations on Transition Boundaries:*
- The first observed non-zero post-stimulus activity occurred at $\alpha = 0.020\,\text{mV / synapse contact}$ in the tested grid.
- Numerical instability was first observed at $\alpha = 0.040\,\text{mV / synapse contact}$ in the tested grid.
- These reflect empirical properties of the tested discrete grid, not exact mathematical bifurcation points.

---

## 4. Corrected Representative Regimes & Limited Lesion Validation

Based strictly on the measured data above, three representative coupling scales were defined:
1. **Low-Coupling Representative ($\alpha = 0.0000\,\text{mV/synapse}$):** Purely uncoupled integrate-and-fire units receiving direct feedforward drive with zero recurrent transmission.
2. **Representative Transition-Region Coupling ($\alpha = 0.0200\,\text{mV/synapse}$):** Coupling scale where recruitment of unstimulated neurons becomes substantial ($146$ neurons, $19.47\%$) and post-stimulus activity first emerges ($0.060\,\text{Hz}$).
3. **Strong-Recurrent Representative ($\alpha = 0.0300\,\text{mV/synapse}$):** Coupling scale where recurrent connectivity recruits $87.07\%$ of unstimulated neurons ($653$ neurons) and sustains post-stimulus reverberation ($1.558\,\text{Hz}$).

A limited lesion validation was conducted across these 3 representative scales ($25\%$ stimulus, $10\%$ and $20\%$ lesions for hub and random across 5 seeds: `42, 123, 456, 789, 1000`, plus intact $0\%$ baselines referenced from the intact sweep):
- **Intact baseline ($0\%$ lesion):** $1$ condition per $\alpha$ directly preserved from the intact sweep ($3$ baseline records, not redundantly re-simulated).
- **Hub lesions ($10\%, 20\%$):** $2$ lesion fractions $\times$ $1$ run $\times$ $3$ $\alpha$ = $6$ simulations.
- **Random lesions ($10\%, 20\%$):** $2$ lesion fractions $\times$ $5$ seeds $\times$ $3$ $\alpha$ = $30$ simulations.
- **Total freshly executed simulations:** $(2 + 10) \times 3 = \mathbf{36}\text{ simulations}$.
- **Total recorded condition rows in table:** $3\text{ intact} + 36\text{ lesion} = \mathbf{39}\text{ rows}$.

Each condition was normalized against the intact baseline at the **same $\alpha$**:

| Representative Regime | $\alpha$ (mV/syn) | Lesion Condition | Edge Retention | Weight Retention | Mean Rate | Activity Robustness ($R_{\text{act}}$) | Recruited Unstimulated | Post-Stim Rate |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low** | $0.0000$ | Intact Baseline | $100.0\%$ | $100.0\%$ | $2.25\,\text{Hz}$ | **$1.0000$** | $0$ ($0.0\%$) | $0.000\,\text{Hz}$ |
| Low | $0.0000$ | Hub 10% | $61.5\%$ | $60.2\%$ | $2.29\,\text{Hz}$ | **$1.0178$** | $0$ ($0.0\%$) | $0.000\,\text{Hz}$ |
| Low | $0.0000$ | Hub 20% | $41.8\%$ | $40.7\%$ | $2.31\,\text{Hz}$ | **$1.0267$** | $0$ ($0.0\%$) | $0.000\,\text{Hz}$ |
| Low | $0.0000$ | Random 20% (Mean) | $64.7\%$ | $64.7\%$ | $2.28 \pm 0.05\,\text{Hz}$ | **$1.0124 \pm 0.0211$** | $0$ ($0.0\%$) | $0.000\,\text{Hz}$ |
| **Transition** | $0.0200$ | Intact Baseline | $100.0\%$ | $100.0\%$ | $4.41\,\text{Hz}$ | **$1.0000$** | $146$ ($19.5\%$) | $0.060\,\text{Hz}$ |
| Transition | $0.0200$ | Hub 10% | $61.5\%$ | $60.2\%$ | $2.36\,\text{Hz}$ | **$0.5351$** | $12$ ($1.8\%$) | $0.000\,\text{Hz}$ |
| Transition | $0.0200$ | Hub 20% | $41.8\%$ | $40.7\%$ | $2.32\,\text{Hz}$ | **$0.5261$** | $2$ ($0.3\%$) | $0.000\,\text{Hz}$ |
| Transition | $0.0200$ | Random 20% (Mean) | $64.7\%$ | $64.7\%$ | $3.17 \pm 0.69\,\text{Hz}$ | **$0.7646 \pm 0.1583$** | $63 \pm 34$ | $0.012 \pm 0.018\,\text{Hz}$ |
| **Strong Recurrent** | $0.0300$ | **Intact Baseline** | $100.0\%$ | $100.0\%$ | **$19.48\,\text{Hz}$** | **$1.0000$** | **$653$ ($87.1\%$)** | **$1.558\,\text{Hz}$** |
| Strong Recurrent | $0.0300$ | **Hub 10%** | $61.5\%$ | $60.2\%$ | **$3.77\,\text{Hz}$** | **$0.1935$** | **$92$ ($13.7\%$)** | $0.000\,\text{Hz}$ |
| Strong Recurrent | $0.0300$ | **Hub 20%** | $41.8\%$ | $40.7\%$ | **$2.99\,\text{Hz}$** | **$0.1535$** | **$42$ ($7.1\%$)** | $0.050\,\text{Hz}$ |
| Strong Recurrent | $0.0300$ | **Random 20% (Mean)** | $64.7\%$ | $64.7\%$ | **$7.85 \pm 1.74\,\text{Hz}$** | **$0.4030 \pm 0.0881$** | **$299 \pm 83$** | $0.091 \pm 0.108\,\text{Hz}$ |

---

## 5. Measured Lesion Responses Across Representative Regimes

### At $\alpha = 0.0000$ (Low Coupling)
- Lesioning does not alter normalized mean firing rate substantially: activity robustness remains at $R_{\text{act}} = 1.018$ (Hub 10%) and $1.027$ (Hub 20%).
- Unstimulated recruitment remains strictly $0$ ($0.0\%$), and post-stimulus rate is $0.000\,\text{Hz}$.
- Firing rate is arithmetically buffered because stimulated neurons fire independently and the rate is normalized by surviving neuron count ($N_{\text{surviving}}$).

### At $\alpha = 0.0200$ (Transition-Region Coupling)
- Lesioning substantially reduces recurrent recruitment and post-stimulus activity:
  - Hub 10% lesion reduces activity robustness to $R_{\text{act}} = 0.5351$, decimating recruited unstimulated neurons from $146 \to 12$ ($1.8\%$) and eliminating post-stimulus firing ($0.000\,\text{Hz}$).
  - Hub 20% lesion reduces activity robustness to $R_{\text{act}} = 0.5261$, reducing recruited unstimulated neurons to just $2$ ($0.3\%$).
  - Random 20% lesions produce an intermediate reduction ($R_{\text{act}} = 0.7646 \pm 0.1583$), retaining $63 \pm 34$ recruited neurons.

### At $\alpha = 0.0300$ (Strong Recurrent Coupling)
- Targeted hub removal strongly suppresses recurrent recruitment and persistent activity:
  - Hub 10% lesion reduces activity robustness to **$R_{\text{act}} = 0.1935$** (an $80.6\%$ reduction), reducing recruited unstimulated neurons from $653 \to 92$ and eliminating post-stimulus firing ($0.000\,\text{Hz}$).
  - Hub 20% lesion reduces activity robustness to **$R_{\text{act}} = 0.1535$** (an $84.6\%$ reduction), reducing recruited unstimulated neurons to $42$ ($7.1\%$).
  - Random 20% lesions produce a smaller, moderate reduction ($R_{\text{act}} = 0.4030 \pm 0.0881$), retaining $299 \pm 83$ recruited neurons.
- Within the tested high-coupling LIF regime, targeted removal of high-degree neurons produced substantially larger reductions in recurrent activity than random removal.

---

## 6. Scientific Conclusion

> **Question: Does recurrent coupling materially influence network activity under any tested $\alpha$?**

**Answer: Yes, within the tested model and $\alpha$ range.**

Across the tested coupling grid, recurrent connectivity had limited influence at $\alpha \le 0.010$, became increasingly influential between $\alpha = 0.015$ and $0.030$, and produced numerical instability at $\alpha \ge 0.040$ under the excitatory-only model. At $\alpha = 0.030$, targeted high-degree neuron lesions substantially reduced recurrent recruitment and post-stimulus activity compared with the intact network.

The Phase 4 robustness result is strongly dependent on the tested coupling regime. At $\alpha = 0.01$, activity is predominantly externally driven and recurrent recruitment is limited. At higher coupling scales, recurrent network dynamics become substantially more influential and lesion sensitivity increases. These findings demonstrate strong dependence of lesion sensitivity on the computational coupling regime, rather than establishing biological resilience or vulnerability of the *Drosophila* brain.

---

## 7. Methodological Limitations

- **Model-Design Origin of Instability:** The model represents all connectome edges as positive synaptic jumps. Consequently, it lacks inhibitory synaptic effects that could oppose recurrent excitation. The observed numerical instability therefore reflects the behavior of this simplified excitatory-only model, rather than biological seizure dynamics.
- **Connectome Partiality:** The 1,000-neuron subnetwork represents a central, high-degree subnetwork of the Janelia Hemibrain and lacks peripheral sensory and motor structures.
- **Coupling Scale ($\alpha$) Interpretation:** $\alpha$ is a computational scalar ($\text{mV / synapse contact}$), not a biophysically calibrated quantal conductance with dynamic reversal potentials ($E_{\text{syn}}$).
- **No Behavioral Equivalence:** Stability or reduction of firing rates reflects computational properties of an integrate-and-fire model under external drive, not biological resilience or behavioral preservation in living flies.
- **Coarse Grid Bounds:** Testing 11 discrete $\alpha$ values identifies an observed transition region ($0.015 \le \alpha < 0.040$), not an exact mathematical bifurcation point.

---

## 8. Generated Outputs & Artifacts

### Data Tables (`results/tables/`)
- [`coupling_sweep_raw_results.csv`](file:///C:/Users/valok/.gemini/antigravity-ide/scratch/fly-connectome-research/results/tables/coupling_sweep_raw_results.csv) (all 55 intact simulations across 11 $\alpha$ values, strictly preserved)
- [`coupling_sweep_summary.csv`](file:///C:/Users/valok/.gemini/antigravity-ide/scratch/fly-connectome-research/results/tables/coupling_sweep_summary.csv) (summary across $\alpha$ and stimulus fractions)
- [`coupling_regimes.csv`](file:///C:/Users/valok/.gemini/antigravity-ide/scratch/fly-connectome-research/results/tables/coupling_regimes.csv) (regime classifications for primary 25% drive)
- [`coupling_lesion_validation.csv`](file:///C:/Users/valok/.gemini/antigravity-ide/scratch/fly-connectome-research/results/tables/coupling_lesion_validation.csv) (36 newly simulated lesion validation runs + 3 intact baseline rows = 39 condition rows across representative $\alpha \in \{0.0000, 0.0200, 0.0300\}$)

### Reproducibility Verification
- An independent verification script (`scratch/verify_lesion_reproducibility.py`) was executed to re-simulate all 36 lesion conditions from scratch with identical seeds.
- **Verification Result:** The two runs produced exactly identical numerical values across all compared result fields (spikes, firing rates, active fractions, graph edge counts, and robustness metrics).

### Publication Figures (`results/figures/` and Artifacts)
- `coupling_mean_firing_rate.png` (Figure 1: Population mean firing rate vs $\alpha$)
- `coupling_post_stimulus_rate.png` (Figure 2: Post-stimulus firing rate vs $\alpha$)
- `coupling_unstimulated_recruitment.png` (Figure 3: Unstimulated neuron recruitment vs $\alpha$)
- `coupling_stim_vs_unstim_firing.png` (Figure 4: Stimulated vs unstimulated firing comparison)
- `coupling_activity_regimes.png` (Figure 5: Observed dynamic regime diagram across $\alpha$)
- `coupling_max_membrane_potential.png` (Figure 6: Maximum membrane potential vs $\alpha$ showing stability boundary)
- `coupling_total_spikes.png` (Figure 7: Total emitted spikes vs $\alpha$)
- `coupling_post_stim_persistence.png` (Figure 8: Post-stimulus persistence ratio vs $\alpha$)
- `coupling_structural_vs_recruitment.png` (Figure 9: Structural recurrent input vs functional neuron recruitment)

### Unit Tests
- [`tests/test_coupling_sweep.py`](file:///C:/Users/valok/.gemini/antigravity-ide/scratch/fly-connectome-research/tests/test_coupling_sweep.py) (10 unit tests, full repo at 54/54 passing).
