# Computational Modeling Assumptions & Biophysical Boundary Conditions

**Project:** Virtual Lesion Analysis of a Drosophila Connectome Subnetwork  
**Module:** Connectome-Constrained Leaky Integrate-and-Fire (LIF) Simulation

---

## 1. Why the Leaky Integrate-and-Fire (LIF) Model Was Selected

Simulating multi-compartment Hodgkin-Huxley or conductance-based biophysical models for thousands of neurons requires extensive sets of unknown ion channel conductances, gating kinetics, and morphology parameters that are currently unmeasured for most Drosophila cell types. 

The standard point **Leaky Integrate-and-Fire (LIF)** model was selected because:
1. **Computational Tractability**: It allows exact vectorized integration of 1,000+ neurons and >100,000 synaptic connections on standard consumer hardware (Intel Core i5, 16 GB RAM) with sub-second execution times.
2. **Topological Focus**: By keeping single-cell intrinsic dynamics minimal and standardized, observed differences in population activity and lesion vulnerability can be directly attributed to **network topology (connectome graph architecture)** rather than arbitrary cell-intrinsic parameter tuning.
3. **Analytic Reproducibility**: The model contains a minimal number of free parameters, facilitating systematic parameter sweeps and sensitivity analyses.

---

## 2. Governing Equations

The subthreshold membrane potential dynamics for each neuron $i \in \{1, \dots, N\}$ are governed by:

$$\tau_m \frac{dV_i(t)}{dt} = -(V_i(t) - V_{\text{rest}}) + R_m \cdot I_{\text{ext}, i}(t) + I_{\text{syn}, i}(t) + \xi_i(t)$$

Where:
- $\tau_m = R_m C_m$ is the membrane time constant.
- $V_i(t)$ is the membrane potential.
- $V_{\text{rest}}$ is the resting membrane potential.
- $I_{\text{ext}, i}(t)$ is the external driving stimulus (injected current).
- $I_{\text{syn}, i}(t)$ is the connectome-derived synaptic current.
- $\xi_i(t)$ represents stochastic Gaussian membrane noise: $\langle \xi_i(t) \xi_j(t') \rangle = 2 D \delta_{ij} \delta(t - t')$.

### Spike Generation & Reset Mechanism
When the membrane potential reaches or exceeds an action potential threshold $V_{\text{threshold}}$:
$$V_i(t) \ge V_{\text{threshold}} \implies \begin{cases} S_i(t) = 1 & \text{(Action Potential Emitted)} \\ V_i(t^+) = V_{\text{reset}} & \text{(Membrane Reset)} \end{cases}$$

Following a spike, the neuron enters an **absolute refractory period** $t_{\text{ref}}$, during which the membrane potential is clamped to $V_{\text{reset}}$ and integration is paused:
$$V_i(t) = V_{\text{reset}} \quad \text{for } t \in [t_{\text{spike}}, t_{\text{spike}} + t_{\text{ref}}]$$

---

## 3. Parameter Values & Justification

| Parameter | Symbol | Value | Unit | Biophysical / Computational Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Membrane Time Constant** | $\tau_m$ | $20.0$ | $\text{ms}$ | Typical passive membrane charging time in invertebrate central neurons. |
| **Resting Potential** | $V_{\text{rest}}$ | $-65.0$ | $\text{mV}$ | Standard intracellular resting potential. |
| **Reset Potential** | $V_{\text{reset}}$ | $-70.0$ | $\text{mV}$ | Hyperpolarized potential following spike repolarization. |
| **Spike Threshold** | $V_{\text{threshold}}$ | $-50.0$ | $\text{mV}$ | Voltage threshold ($15\,\text{mV}$ above rest). |
| **Refractory Period** | $t_{\text{ref}}$ | $2.0$ | $\text{ms}$ | Inactivation window of voltage-gated sodium/calcium channels. |
| **Integration Timestep** | $dt$ | $0.5$ | $\text{ms}$ | Numerically stable discretization for $\tau_m = 20\,\text{ms}$ ($dt \ll \tau_m$). |
| **Synaptic Weight Scale** | $\alpha$ | $0.01$ | $\text{mV/synapse}$ | Effective postsynaptic depolarization per anatomical presynaptic contact site. |
| **External Current** | $I_{\text{ext}}$ | $18.0$ | $\text{mV}$ | Suprathreshold driving current injected into sensory/input fraction. |

---

## 4. Why Synaptic Scaling Is Required

In raw connectomics, edge weights represent **discrete anatomical synapse counts** (the number of presynaptic active zones / T-bars and postsynaptic densities reconstructed from electron microscopy). 

However:
1. **Raw synapse counts are not electrical currents**: A synapse count of $w = 20$ indicates 20 physical contact sites, not an injection of $20\,\text{mV}$ or $20\,\text{pA}$.
2. **Dense Recurrent Convergence**: In our 1,000-neuron subnetwork, the average in-degree is $\langle k_{in} \rangle \approx 113$ connections, with total incoming synapse counts averaging $\approx 937$ per neuron.
3. **Runaway Epileptiform Avalanches**: If unscaled weights ($\alpha \ge 0.05$) are applied directly, a single presynaptic volley causes massive, non-physiological recurrent depolarization ($> 120\,\text{mV}$), driving the network into numerical instability.

Therefore, an explicit scaling parameter is introduced:
$$W_{\text{effective}, ji} = \alpha \times W_{\text{raw}, ji}$$

Our empirical parameter sweep demonstrates that values in the range $\alpha \in [0.005, 0.010]$ establish stable, sparse baseline firing rates ($\approx 2.2 - 2.3\,\text{Hz}$) under external drive without runaway excitation. Here, $\alpha$ serves as a computational scaling parameter rather than an empirically calibrated biological synaptic efficacy.

---

## 5. Structural vs. Functional Connectivity

- **Structural Connectivity (Empirical Connectome)**: Physical anatomical wiring diagrams reconstructed via volume electron microscopy (EM). Captures who *can* communicate.
- **Functional / Effective Connectivity**: Temporal correlations and causal influence during active neural processing. Captures who *is* communicating.
- **Connectome Limitations**:
  - Does not measure neurotransmitter receptor subtypes (e.g. nicotinic vs muscarinic, ionotropic vs metabotropic).
  - Omits electrical gap junctions (innexins) not resolved in standard chemical synapse annotations.
  - Omits volume transmission (paracrine signaling via dopamine, octopamine, serotonin, and neuropeptides).

---

## 6. Subnetwork Sampling Bias (Highest-Degree Selection)

Our baseline 1,000-neuron network was extracted using the **highest-degree** strategy:
1. **Why Selected**: It forms a cohesive, strongly connected core ($S = 1.0$, density $= 0.113$, 113,089 directed edges) that ensures signal propagation across the network without immediately dissipating into isolated components.
2. **Sampling Bias**:
   - This subnetwork is **not** a representative random cross-section of the whole fly brain.
   - It is enriched for large multi-neuropil projection neurons, central complex integrator hubs, and broad interneurons.
   - Peripheral sensory inputs and sparse feedforward layers are underrepresented.
   - Consequently, graph density and recurrence are significantly higher than the whole-brain average ($0.113$ vs $0.009$).

---

## 7. Scientific Boundaries & Non-Claims

> [!CAUTION]
> - This simulation **does not** reproduce the cognitive behavior, sensory perception, or motor outputs of a live fruit fly.
> - The results describe **the dynamical consequences of connectome graph topology under a standardized point-neuron model**.
> - Any observed shifts in firing rate or stability following virtual lesions reflect **structural vulnerability of the wiring diagram**, which forms a hypothesis for future experimental validation in biological organisms.
