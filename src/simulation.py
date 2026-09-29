"""
Connectome-Constrained Leaky Integrate-and-Fire (LIF) Simulation.

Scientific Disclaimer:
The connectome provides empirical structural connectivity (anatomical synapse counts).
The Leaky Integrate-and-Fire (LIF) parameters represent computational modeling simplifications
and do not claim to reproduce the full biophysical or behavioral complexity of live Drosophila.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from src.data_loader import SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class LIFConfig:
    """
    Configuration parameters for connectome-constrained LIF simulation.

    Attributes:
        tau_m: Membrane time constant in milliseconds (default: 20.0 ms).
        v_rest: Resting membrane potential in mV (default: -65.0 mV).
        v_reset: Post-spike reset membrane potential in mV (default: -70.0 mV).
        v_threshold: Action potential spike threshold in mV (default: -50.0 mV).
        t_ref: Absolute refractory period in milliseconds (default: 2.0 ms).
        dt: Euler numerical integration step in milliseconds (default: 0.5 ms).
        duration: Total simulation time in milliseconds (default: 1000.0 ms).
        synaptic_weight_scale: Scaling factor mapping anatomical synapse counts
            to effective post-synaptic potential change in mV (default: 0.01).
        external_stimulus_mode: Stimulus protocol: 'constant', 'random', or 'pulse'.
        external_current: Injected current amplitude (in mV/ms equivalent, default: 12.0).
        noise_sigma: Standard deviation of Gaussian background membrane noise (default: 1.0).
        pulse_start: Pulse onset time in milliseconds (default: 200.0 ms).
        pulse_end: Pulse offset time in milliseconds (default: 600.0 ms).
        stimulus_fraction: Fraction of neurons receiving direct external stimulus (default: 0.25).
        random_seed: Random seed for deterministic reproducibility (default: 42).
        max_rate_threshold: Firing rate threshold (Hz) above which network is flagged unstable (default: 350.0).
    """

    tau_m: float = 20.0
    v_rest: float = -65.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    t_ref: float = 2.0
    dt: float = 0.5
    duration: float = 1000.0
    synaptic_weight_scale: float = 0.01
    external_stimulus_mode: str = "constant"
    external_current: float = 12.0
    noise_sigma: float = 1.0
    pulse_start: float = 200.0
    pulse_end: float = 600.0
    stimulus_fraction: float = 0.25
    random_seed: int = 42
    max_rate_threshold: float = 350.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    """
    Standardized simulation output container.
    """

    spike_times: List[Tuple[float, int]]  # (time_ms, neuron_id)
    spike_counts: np.ndarray  # Shape: (N,)
    firing_rates: np.ndarray  # Shape: (N,), in Hz
    time_vector: np.ndarray  # Shape: (T,)
    population_rate: np.ndarray  # Shape: (num_bins,), in spikes/sec
    bin_centers: np.ndarray  # Shape: (num_bins,)
    neuron_ids: List[int]  # Mapping from internal index to graph neuron ID
    membrane_samples: Optional[np.ndarray] = None  # Shape: (num_samples, T)
    sample_neuron_ids: Optional[List[int]] = None
    is_stable: bool = True
    status_message: str = "Simulation completed normally."
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Optional[LIFConfig] = None


class LIFNetwork:
    """
    Efficient vectorized Leaky Integrate-and-Fire network driven by real connectome topology.
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Initializes network structures from a NetworkX directed connectome.

        Parameters:
            graph: Directed connectome graph with 'weight' edge attributes.
        """
        self.graph = graph
        self.neuron_ids = sorted(list(graph.nodes()))
        self.n_neurons = len(self.neuron_ids)
        self.id_to_idx = {nid: idx for idx, nid in enumerate(self.neuron_ids)}

        # Build sparse weight matrix W[pre, post]
        # In matrix algebra: input_to_post = W.T @ spikes_pre
        if self.n_neurons > 0:
            sources = []
            targets = []
            weights = []
            for u, v, data in graph.edges(data=True):
                if u in self.id_to_idx and v in self.id_to_idx:
                    sources.append(self.id_to_idx[u])
                    targets.append(self.id_to_idx[v])
                    weights.append(float(data.get("weight", 1.0)))

            if sources:
                self.W = sparse.csr_matrix(
                    (weights, (sources, targets)),
                    shape=(self.n_neurons, self.n_neurons),
                    dtype=np.float32,
                )
            else:
                self.W = sparse.csr_matrix((self.n_neurons, self.n_neurons), dtype=np.float32)
        else:
            self.W = sparse.csr_matrix((0, 0), dtype=np.float32)

    def simulate(
        self,
        config: LIFConfig,
        record_samples: int = 5,
        stimulated_neuron_ids: Optional[Sequence[int]] = None,
    ) -> SimulationResult:
        """
        Executes vectorized numerical integration of LIF dynamics over the connectome.

        Parameters:
            config: Simulation configuration parameters.
            record_samples: Number of sample neurons to record continuous membrane potential for.
            stimulated_neuron_ids: Explicit list of pre-lesion neuron IDs to receive direct external drive.
                Surviving neurons with these IDs are stimulated; ablated neurons naturally receive no drive.

        Returns:
            SimulationResult containing spike rasters, rates, and diagnostics.
        """
        if self.n_neurons == 0:
            return SimulationResult(
                spike_times=[],
                spike_counts=np.array([], dtype=int),
                firing_rates=np.array([], dtype=float),
                time_vector=np.array([], dtype=float),
                population_rate=np.array([], dtype=float),
                bin_centers=np.array([], dtype=float),
                neuron_ids=[],
                is_stable=True,
                status_message="Empty network.",
                config=config,
            )

        rng = np.random.default_rng(config.random_seed)

        # Simulation time discretization
        n_steps = int(np.round(config.duration / config.dt))
        time_vector = np.linspace(0.0, config.duration, n_steps, endpoint=False, dtype=np.float32)
        refractory_steps = int(np.round(config.t_ref / config.dt))

        # State vectors
        V = np.full(self.n_neurons, config.v_rest, dtype=np.float32)
        refractory_counter = np.zeros(self.n_neurons, dtype=np.int32)

        # External stimulus mask (which neurons receive direct stimulation)
        stim_mask = np.zeros(self.n_neurons, dtype=bool)
        if stimulated_neuron_ids is not None:
            # Map pre-selected neuron IDs to surviving indices
            for nid in stimulated_neuron_ids:
                if nid in self.id_to_idx:
                    stim_mask[self.id_to_idx[nid]] = True
        else:
            if config.stimulus_fraction > 0:
                n_stimulated = max(1, int(np.round(self.n_neurons * config.stimulus_fraction)))
                stimulated_indices = rng.choice(self.n_neurons, size=n_stimulated, replace=False)
                stim_mask[stimulated_indices] = True


        # Pre-scale synaptic weight matrix: W_eff[pre, post]
        effective_W_T = (self.W.T * float(config.synaptic_weight_scale)).tocsr()

        # Recording structures
        spike_times: List[Tuple[float, int]] = []
        spike_counts = np.zeros(self.n_neurons, dtype=np.int32)

        # Sample membrane traces
        sample_indices: List[int] = []
        if record_samples > 0:
            sample_indices = list(range(min(record_samples, self.n_neurons)))
            membrane_samples = np.zeros((len(sample_indices), n_steps), dtype=np.float32)
        else:
            membrane_samples = None

        # Instability detection flags
        is_stable = True
        status_message = "Simulation completed normally."

        # Integration constants
        decay = float(np.exp(-config.dt / config.tau_m))
        inv_tau = float(config.dt / config.tau_m)
        noise_scale = float(config.noise_sigma * np.sqrt(config.dt))

        v_max_observed = float(V.max()) if self.n_neurons > 0 else float(config.v_rest)
        v_min_observed = float(V.min()) if self.n_neurons > 0 else float(config.v_rest)

        # Main Time-Stepping Loop (vectorized over all N neurons)
        for step in range(n_steps):
            t_curr = float(time_vector[step])

            # 1. Determine external current for this timestep
            I_ext = np.zeros(self.n_neurons, dtype=np.float32)
            mode = config.external_stimulus_mode.lower()

            if mode == "constant":
                I_ext[stim_mask] = config.external_current
            elif mode == "pulse":
                if config.pulse_start <= t_curr < config.pulse_end:
                    I_ext[stim_mask] = config.external_current
                else:
                    I_ext[stim_mask] = 0.0
            elif mode == "random":
                # Background fluctuating current
                I_ext[stim_mask] = config.external_current + rng.standard_normal(n_stimulated).astype(np.float32) * config.noise_sigma
            else:
                I_ext[stim_mask] = config.external_current

            # Add continuous membrane noise
            if config.noise_sigma > 0:
                noise = rng.standard_normal(self.n_neurons).astype(np.float32) * noise_scale
            else:
                noise = np.zeros(self.n_neurons, dtype=np.float32)

            # 2. Subthreshold Integration for non-refractory neurons
            active_mask = refractory_counter == 0

            # dV/dt = -(V - v_rest) + I_ext + noise
            # Analytical / Euler step
            V[active_mask] = (
                config.v_rest + (V[active_mask] - config.v_rest) * decay
                + (I_ext[active_mask] + noise[active_mask]) * inv_tau
            )

            # Neurons in refractory period remain clamped
            V[~active_mask] = config.v_reset
            refractory_counter[~active_mask] -= 1

            # 3. Detect Spikes: V >= v_threshold
            spiking_mask = (V >= config.v_threshold) & active_mask
            num_spikes_now = int(spiking_mask.sum())

            if num_spikes_now > 0:
                spiking_indices = np.where(spiking_mask)[0]
                spike_counts[spiking_indices] += 1

                for idx in spiking_indices:
                    spike_times.append((t_curr, self.neuron_ids[idx]))

                # Reset membrane potential and activate refractory counter
                V[spiking_indices] = config.v_reset
                refractory_counter[spiking_indices] = refractory_steps

                # 4. Synaptic Transmission to postsynaptic targets
                # I_syn = effective_W_T @ spikes_vector
                syn_input = effective_W_T.dot(spiking_mask.astype(np.float32))
                # Deliver EPSP/IPSP directly into non-refractory membrane
                V[active_mask] += syn_input[active_mask]

            if self.n_neurons > 0:
                v_max_observed = max(v_max_observed, float(np.max(V)))
                v_min_observed = min(v_min_observed, float(np.min(V)))

            # Record membrane traces for sample neurons
            if membrane_samples is not None:
                membrane_samples[:, step] = V[sample_indices]

            # 5. Continuous Sanity Checks & Instability Detection
            if np.isnan(V).any() or np.isinf(V).any():
                is_stable = False
                status_message = "Simulation unstable under current parameterization: NaN/Inf detected in membrane potential."
                logger.warning(status_message)
                break

            if (V > 120.0).any():
                is_stable = False
                status_message = f"Simulation unstable under current parameterization: Membrane potential exploded (>120 mV) at t={t_curr:.1f} ms."
                logger.warning(status_message)
                break

        # Calculate firing rate statistics
        duration_sec = config.duration / 1000.0
        firing_rates = spike_counts / duration_sec if duration_sec > 0 else np.zeros_like(spike_counts, dtype=float)

        mean_rate = float(firing_rates.mean())
        max_rate = float(firing_rates.max())
        median_rate = float(np.median(firing_rates))
        active_fraction = float((spike_counts > 0).sum() / self.n_neurons)
        total_spikes = int(spike_counts.sum())

        if max_rate > config.max_rate_threshold:
            is_stable = False
            status_message = (
                f"Simulation unstable under current parameterization: Max firing rate ({max_rate:.1f} Hz) "
                f"exceeds biological stability limit ({config.max_rate_threshold:.1f} Hz)."
            )
            logger.warning(status_message)

        # Compute Population Binned Firing Rate
        pop_rate, bin_centers = compute_population_rate(
            spike_times=spike_times,
            num_neurons=self.n_neurons,
            duration=config.duration,
            bin_width_ms=10.0,
        )

        metrics = {
            "total_spikes": total_spikes,
            "spikes_per_neuron": round(total_spikes / self.n_neurons, 2),
            "mean_firing_rate_hz": round(mean_rate, 2),
            "median_firing_rate_hz": round(median_rate, 2),
            "max_firing_rate_hz": round(max_rate, 2),
            "firing_rate_variance": round(float(np.var(firing_rates)), 2),
            "active_neuron_fraction": round(active_fraction, 4),
            "v_max_observed": round(v_max_observed, 2),
            "v_min_observed": round(v_min_observed, 2),
            "is_stable": is_stable,
            "status_message": status_message,
        }

        sample_ids = [self.neuron_ids[idx] for idx in sample_indices] if sample_indices else None

        return SimulationResult(
            spike_times=spike_times,
            spike_counts=spike_counts,
            firing_rates=firing_rates,
            time_vector=time_vector,
            population_rate=pop_rate,
            bin_centers=bin_centers,
            neuron_ids=self.neuron_ids,
            membrane_samples=membrane_samples,
            sample_neuron_ids=sample_ids,
            is_stable=is_stable,
            status_message=status_message,
            metrics=metrics,
            config=config,
        )


def compute_population_rate(
    spike_times: Sequence[Tuple[float, int]],
    num_neurons: int,
    duration: float,
    bin_width_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes binned population spike rate in spikes/second (Hz equivalent).

    Parameters:
        spike_times: List of (time_ms, neuron_id) tuples.
        num_neurons: Total neuron count in population.
        duration: Total duration in milliseconds.
        bin_width_ms: Histogram bin width in milliseconds.

    Returns:
        (population_rate_array, bin_centers_array)
    """
    if duration <= 0 or bin_width_ms <= 0 or num_neurons == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_bins = int(np.ceil(duration / bin_width_ms))
    bins = np.linspace(0.0, duration, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    if not spike_times:
        return np.zeros(n_bins, dtype=float), bin_centers

    times = [t for t, _ in spike_times]
    counts, _ = np.histogram(times, bins=bins)

    # Convert counts per bin to spikes per second per neuron (Hz)
    bin_width_sec = bin_width_ms / 1000.0
    rate = counts / (num_neurons * bin_width_sec)
    return rate, bin_centers


def run_simulation(
    graph: nx.DiGraph,
    config: Optional[LIFConfig] = None,
    stimulated_neuron_ids: Optional[Sequence[int]] = None,
    **kwargs,
) -> SimulationResult:
    """
    Convenience function to run a connectome-constrained LIF simulation.

    Parameters:
        graph: DiGraph with synaptic weights.
        config: LIFConfig instance (optional).
        stimulated_neuron_ids: Optional pre-selected neuron IDs to receive external stimulation.
        **kwargs: Overrides for LIFConfig parameters.
    """
    if config is None:
        config = LIFConfig(**kwargs)
    elif kwargs:
        cfg_dict = config.to_dict()
        cfg_dict.update(kwargs)
        config = LIFConfig(**cfg_dict)

    network = LIFNetwork(graph)
    return network.simulate(config, stimulated_neuron_ids=stimulated_neuron_ids)


def select_stimulated_neurons(
    graph: nx.DiGraph,
    stimulus_fraction: float,
    seed: int = 42,
) -> List[int]:
    """
    Deterministically selects exactly round(N * stimulus_fraction) neurons from an intact graph.
    Selection is made on the intact pre-lesion network.

    Parameters:
        graph: Intact connectome DiGraph.
        stimulus_fraction: Proportion of neurons to stimulate (e.g. 0.25, 0.10, 0.05, 0.01, 0.0).
        seed: Random seed for deterministic selection.

    Returns:
        Sorted list of selected neuron IDs.
    """
    total_neurons = graph.number_of_nodes()
    n_stim = int(np.round(total_neurons * stimulus_fraction))
    if n_stim <= 0 or total_neurons == 0:
        return []

    sorted_nodes = np.array(sorted(list(graph.nodes())))
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(sorted_nodes), size=n_stim, replace=False)
    selected_nodes = sorted_nodes[selected_indices].tolist()
    return sorted(selected_nodes)


def compute_temporal_metrics(
    spike_times: Sequence[Tuple[float, int]],
    num_neurons: int,
    stimulus_window: Tuple[float, float] = (200.0, 600.0),
    post_window: Tuple[float, float] = (600.0, 1000.0),
) -> Dict[str, Any]:
    """
    Computes spike counts and firing rates separated into stimulus and post-stimulus windows.
    Safely handles division-by-zero without producing infinity.

    Parameters:
        spike_times: List of (time_ms, neuron_id) tuples.
        num_neurons: Number of surviving neurons in network.
        stimulus_window: (start_ms, end_ms) for external drive epoch.
        post_window: (start_ms, end_ms) for post-stimulation decay epoch.

    Returns:
        Dictionary containing temporal metrics and ratio.
    """
    stim_start, stim_end = stimulus_window
    post_start, post_end = post_window
    stim_dur_sec = (stim_end - stim_start) / 1000.0
    post_dur_sec = (post_end - post_start) / 1000.0

    stim_spikes = sum(1 for t, _ in spike_times if stim_start <= t < stim_end)
    post_spikes = sum(1 for t, _ in spike_times if post_start <= t <= post_end)

    stim_rate = (stim_spikes / (num_neurons * stim_dur_sec)) if (num_neurons > 0 and stim_dur_sec > 0) else 0.0
    post_rate = (post_spikes / (num_neurons * post_dur_sec)) if (num_neurons > 0 and post_dur_sec > 0) else 0.0

    if post_rate > 0:
        stim_to_post_ratio = float(stim_rate / post_rate)
    else:
        # Documented convention: NaN if post-stimulus activity is 0, avoiding infinity
        stim_to_post_ratio = float("nan")

    return {
        "stimulus_window_spikes": stim_spikes,
        "stimulus_window_rate_hz": round(stim_rate, 4),
        "post_stimulus_window_spikes": post_spikes,
        "post_stimulus_window_rate_hz": round(post_rate, 4),
        "stimulus_to_post_ratio": round(stim_to_post_ratio, 4) if not np.isnan(stim_to_post_ratio) else np.nan,
    }

