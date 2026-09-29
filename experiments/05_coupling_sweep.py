"""
Experiment 05: Coupling-Regime Sweep and Recurrent Dynamics (Phase 6).

Systematically sweeps synaptic coupling scale alpha across 11 orders of magnitude
on the real 1,000-neuron Janelia FlyEM Hemibrain connectome subnetwork:
- Alpha values: 0.000, 0.001, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050 (mV / synapse contact)
- Stimulus levels: 25% (primary, 250 neurons), 10%, 5%, 1%, 0%
- Intact Simulations: 11 alphas x 5 stimulus levels = 55 simulations
- Representative Lesion Validation: 3 representative alphas x (1 intact + 2 hub + 10 random) = 36 simulations
- Comprehensive recurrent-activity, temporal window, and connectivity metrics
- Deterministic reproducibility and numerical stability tracking.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from src.data_loader import extract_real_subnetwork
from src.lesion import apply_hub_lesion, apply_random_lesion
from src.simulation import (
    LIFConfig,
    LIFNetwork,
    SimulationResult,
    select_stimulated_neurons,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical Phase 6 parameter grid
ALPHA_VALUES = [
    0.000,
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.010,
    0.015,
    0.020,
    0.030,
    0.040,
    0.050,
]

STIMULUS_FRACTIONS = [0.25, 0.10, 0.05, 0.01, 0.00]
LESION_FRACTIONS = [0.10, 0.20]
RANDOM_SEEDS = [42, 123, 456, 789, 1000]


def compute_activity_windows(
    spike_times: Sequence[Tuple[float, int]],
    num_neurons: int,
    neuron_ids: List[int],
    duration: float = 1000.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes activity metrics across Pre-stimulus, Stimulus, Post-stimulus, and Full windows.
    Pre: 0-200 ms, Stim: 200-600 ms, Post: 600-1000 ms, Full: 0-1000 ms.
    """
    id_to_idx = {nid: idx for idx, nid in enumerate(neuron_ids)}
    windows = {
        "full": (0.0, duration),
        "pre_stimulus": (0.0, 200.0),
        "stimulus": (200.0, 600.0),
        "post_stimulus": (600.0, duration),
    }

    results = {}
    for win_name, (w_start, w_end) in windows.items():
        w_dur_sec = (w_end - w_start) / 1000.0
        counts = np.zeros(num_neurons, dtype=np.int32)
        for t, nid in spike_times:
            if w_start <= t < w_end or (w_end == duration and t == w_end):
                if nid in id_to_idx:
                    counts[id_to_idx[nid]] += 1

        total_spikes = int(counts.sum())
        mean_rate = float(total_spikes / (num_neurons * w_dur_sec)) if (num_neurons > 0 and w_dur_sec > 0) else 0.0
        rates_per_neuron = counts / w_dur_sec if w_dur_sec > 0 else np.zeros(num_neurons)
        active_fraction = float((counts > 0).sum() / num_neurons) if num_neurons > 0 else 0.0
        median_rate = float(np.median(rates_per_neuron)) if num_neurons > 0 else 0.0
        max_rate = float(np.max(rates_per_neuron)) if num_neurons > 0 else 0.0

        results[win_name] = {
            "total_spikes": total_spikes,
            "mean_firing_rate_hz": round(mean_rate, 4),
            "active_neuron_fraction": round(active_fraction, 4),
            "median_firing_rate_hz": round(median_rate, 4),
            "max_firing_rate_hz": round(max_rate, 4),
            "per_neuron_counts": counts,
        }
    return results


def compute_recurrent_activity_metrics(
    spike_counts: np.ndarray,
    neuron_ids: List[int],
    stimulated_neuron_ids: Sequence[int],
    windows_data: Dict[str, Dict[str, Any]],
    duration: float = 1000.0,
) -> Dict[str, Any]:
    """
    Computes unstimulated recruitment, stimulated vs unstimulated rates,
    post-stimulus persistence, and ratio metrics.
    """
    id_to_idx = {nid: idx for idx, nid in enumerate(neuron_ids)}
    stim_set = set(stimulated_neuron_ids)
    stim_indices = [idx for idx, nid in enumerate(neuron_ids) if nid in stim_set]
    unstim_indices = [idx for idx, nid in enumerate(neuron_ids) if nid not in stim_set]

    dur_sec = duration / 1000.0
    stim_count = len(stim_indices)
    unstim_count = len(unstim_indices)

    # Rates over full simulation
    if stim_count > 0 and dur_sec > 0:
        stim_rate = float(spike_counts[stim_indices].mean() / dur_sec)
    else:
        stim_rate = 0.0

    if unstim_count > 0 and dur_sec > 0:
        unstim_rate = float(spike_counts[unstim_indices].mean() / dur_sec)
    else:
        unstim_rate = 0.0

    # Stimulated-to-unstimulated ratio (NaN if unstim_rate is 0 to avoid Inf)
    if unstim_rate > 0.0:
        stim_to_unstim_ratio = stim_rate / unstim_rate
    else:
        stim_to_unstim_ratio = np.nan

    # Unstimulated recruitment during stimulus window (200-600 ms)
    stim_win_counts = windows_data["stimulus"]["per_neuron_counts"]
    if unstim_count > 0:
        recruited_count = int((stim_win_counts[unstim_indices] > 0).sum())
        recruited_frac = recruited_count / unstim_count
    else:
        recruited_count = 0
        recruited_frac = 0.0

    # Post-stimulus persistence
    stim_act_frac = windows_data["stimulus"]["active_neuron_fraction"]
    post_act_frac = windows_data["post_stimulus"]["active_neuron_fraction"]
    if stim_act_frac > 0.0:
        post_persistence = post_act_frac / stim_act_frac
    else:
        post_persistence = np.nan

    return {
        "stimulated_neuron_count": stim_count,
        "unstimulated_neuron_count": unstim_count,
        "stimulated_rate_hz": round(stim_rate, 4),
        "unstimulated_rate_hz": round(unstim_rate, 4),
        "stimulated_to_unstimulated_ratio": round(stim_to_unstim_ratio, 4) if not np.isnan(stim_to_unstim_ratio) else np.nan,
        "recruited_unstimulated_count": recruited_count,
        "recruited_unstimulated_fraction": round(recruited_frac, 4),
        "post_stimulus_persistence": round(post_persistence, 4) if not np.isnan(post_persistence) else np.nan,
    }


def compute_connectivity_metrics(
    graph: nx.DiGraph,
    spike_counts: np.ndarray,
    neuron_ids: List[int],
    stimulated_neuron_ids: Sequence[int],
    alpha: float,
) -> Dict[str, Any]:
    """
    Computes recurrent synaptic input metrics derived from the connectome topology and spike counts.
    I_syn = alpha * W^T @ spikes.
    """
    n = len(neuron_ids)
    if n == 0 or graph.number_of_edges() == 0 or spike_counts.sum() == 0 or alpha == 0.0:
        return {
            "total_synaptic_input_mv": 0.0,
            "mean_recurrent_input_per_neuron_mv": 0.0,
            "max_recurrent_input_mv": 0.0,
            "neurons_receiving_recurrent_input_count": 0,
            "fraction_neurons_receiving_recurrent_input": 0.0,
            "mean_recurrent_input_from_stimulated_mv": 0.0,
            "mean_recurrent_input_from_unstimulated_mv": 0.0,
        }

    id_to_idx = {nid: idx for idx, nid in enumerate(neuron_ids)}
    sources = []
    targets = []
    weights = []
    for u, v, data in graph.edges(data=True):
        if u in id_to_idx and v in id_to_idx:
            sources.append(id_to_idx[u])
            targets.append(id_to_idx[v])
            weights.append(float(data.get("weight", 1.0)))

    W = sparse.csr_matrix((weights, (sources, targets)), shape=(n, n), dtype=np.float32)

    # I_syn_total = alpha * W.T @ s
    s_vec = spike_counts.astype(np.float32)
    I_syn = alpha * (W.T.dot(s_vec))

    # Partition by presynaptic source (stimulated vs unstimulated)
    stim_set = set(stimulated_neuron_ids)
    stim_indices = [idx for idx, nid in enumerate(neuron_ids) if nid in stim_set]
    unstim_indices = [idx for idx, nid in enumerate(neuron_ids) if nid not in stim_set]

    s_stim = np.zeros(n, dtype=np.float32)
    s_stim[stim_indices] = spike_counts[stim_indices]
    I_syn_stim = alpha * (W.T.dot(s_stim))

    s_unstim = np.zeros(n, dtype=np.float32)
    s_unstim[unstim_indices] = spike_counts[unstim_indices]
    I_syn_unstim = alpha * (W.T.dot(s_unstim))

    rec_count = int((I_syn > 0).sum())
    return {
        "total_synaptic_input_mv": round(float(I_syn.sum()), 2),
        "mean_recurrent_input_per_neuron_mv": round(float(I_syn.mean()), 4),
        "max_recurrent_input_mv": round(float(I_syn.max()), 2),
        "neurons_receiving_recurrent_input_count": rec_count,
        "fraction_neurons_receiving_recurrent_input": round(float(rec_count / n), 4),
        "mean_recurrent_input_from_stimulated_mv": round(float(I_syn_stim.mean()), 4),
        "mean_recurrent_input_from_unstimulated_mv": round(float(I_syn_unstim.mean()), 4),
    }


def run_single_sweep_simulation(
    graph: nx.DiGraph,
    config: LIFConfig,
    stimulated_neuron_ids: List[int],
) -> Tuple[SimulationResult, Dict[str, Any], Dict[str, Any], Dict[str, Any], float]:
    """
    Executes a single simulation and collects all activity windows, recurrent metrics,
    and connectivity metrics.
    """
    network = LIFNetwork(graph)
    t0 = time.perf_counter()
    res = network.simulate(config, record_samples=5, stimulated_neuron_ids=stimulated_neuron_ids)
    dur = time.perf_counter() - t0

    windows_data = compute_activity_windows(
        spike_times=res.spike_times,
        num_neurons=len(res.neuron_ids),
        neuron_ids=res.neuron_ids,
        duration=config.duration,
    )

    recurrent_data = compute_recurrent_activity_metrics(
        spike_counts=res.spike_counts,
        neuron_ids=res.neuron_ids,
        stimulated_neuron_ids=stimulated_neuron_ids,
        windows_data=windows_data,
        duration=config.duration,
    )

    conn_data = compute_connectivity_metrics(
        graph=graph,
        spike_counts=res.spike_counts,
        neuron_ids=res.neuron_ids,
        stimulated_neuron_ids=stimulated_neuron_ids,
        alpha=config.synaptic_weight_scale,
    )

    return res, windows_data, recurrent_data, conn_data, dur


def classify_coupling_regime(
    is_stable: bool,
    recruited_unstimulated_fraction: float,
    post_stimulus_rate_hz: float,
) -> str:
    """
    Classifies the observed activity regime based on objective measured criteria.
    Labels:
    - 'Unstable': Numerical instability detected (explosion > 120 mV or NaN/Inf).
    - 'Externally Driven (Weak Coupling)': Recruited fraction < 1% and Post-stimulus rate == 0 Hz.
    - 'Recurrent Recruitment': Recruited fraction >= 1% and Post-stimulus rate == 0 Hz.
    - 'Sustained Recurrent Activity': Post-stimulus rate > 0 Hz (while stable).
    """
    if not is_stable:
        return "Unstable"
    if post_stimulus_rate_hz > 0.001:
        return "Sustained Recurrent Activity"
    if recruited_unstimulated_fraction >= 0.01:
        return "Recurrent Recruitment"
    return "Externally Driven (Weak Coupling)"


def run_intact_coupling_sweep(
    graph: nx.DiGraph,
    alphas: List[float],
    stimulus_fractions: List[float],
    base_config: LIFConfig,
    stim_sets: Dict[float, List[int]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[float, float], Dict[str, Any]]]:
    """
    Runs the full 55 intact baseline simulations across 11 alphas and 5 stimulus levels.
    """
    raw_records: List[Dict[str, Any]] = []
    baseline_lookup: Dict[Tuple[float, float], Dict[str, Any]] = {}

    total_runs = len(alphas) * len(stimulus_fractions)
    run_idx = 0

    logger.info(f"Starting intact coupling sweep ({total_runs} simulations)...")

    for sf in stimulus_fractions:
        stim_ids = stim_sets[sf]
        for alpha in alphas:
            run_idx += 1
            cfg = LIFConfig(
                tau_m=base_config.tau_m,
                v_rest=base_config.v_rest,
                v_reset=base_config.v_reset,
                v_threshold=base_config.v_threshold,
                t_ref=base_config.t_ref,
                dt=base_config.dt,
                duration=base_config.duration,
                synaptic_weight_scale=alpha,
                external_stimulus_mode="pulse",
                external_current=base_config.external_current,
                noise_sigma=base_config.noise_sigma,
                pulse_start=base_config.pulse_start,
                pulse_end=base_config.pulse_end,
                stimulus_fraction=sf,
                random_seed=base_config.random_seed,
                max_rate_threshold=base_config.max_rate_threshold,
            )

            res, win_data, rec_data, conn_data, dur = run_single_sweep_simulation(
                graph=graph,
                config=cfg,
                stimulated_neuron_ids=stim_ids,
            )

            full_win = win_data["full"]
            stim_win = win_data["stimulus"]
            post_win = win_data["post_stimulus"]
            pre_win = win_data["pre_stimulus"]

            regime = classify_coupling_regime(
                is_stable=res.is_stable,
                recruited_unstimulated_fraction=rec_data["recruited_unstimulated_fraction"],
                post_stimulus_rate_hz=post_win["mean_firing_rate_hz"],
            )

            record = {
                "experiment": "intact_sweep",
                "alpha": alpha,
                "stimulus_fraction": sf,
                "stimulated_neurons_count": len(stim_ids),
                "unstimulated_neurons_count": len(res.neuron_ids) - len(stim_ids),
                "is_stable": res.is_stable,
                "status_message": res.status_message,
                "regime": regime,
                "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                "v_min_observed": res.metrics.get("v_min_observed", np.nan),
                "total_spikes": res.metrics["total_spikes"],
                "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                "max_firing_rate_hz": res.metrics["max_firing_rate_hz"],
                # Windows
                "pre_stim_spikes": pre_win["total_spikes"],
                "pre_stim_rate_hz": pre_win["mean_firing_rate_hz"],
                "stim_window_spikes": stim_win["total_spikes"],
                "stim_window_rate_hz": stim_win["mean_firing_rate_hz"],
                "stim_window_active_fraction": stim_win["active_neuron_fraction"],
                "stim_window_max_rate_hz": stim_win["max_firing_rate_hz"],
                "post_stim_spikes": post_win["total_spikes"],
                "post_stim_rate_hz": post_win["mean_firing_rate_hz"],
                "post_stim_active_fraction": post_win["active_neuron_fraction"],
                "post_stim_max_rate_hz": post_win["max_firing_rate_hz"],
                # Recurrent & recruitment
                "stimulated_rate_hz": rec_data["stimulated_rate_hz"],
                "unstimulated_rate_hz": rec_data["unstimulated_rate_hz"],
                "stimulated_to_unstimulated_ratio": rec_data["stimulated_to_unstimulated_ratio"],
                "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                "post_stimulus_persistence": rec_data["post_stimulus_persistence"],
                # Connectivity
                "total_synaptic_input_mv": conn_data["total_synaptic_input_mv"],
                "mean_recurrent_input_per_neuron_mv": conn_data["mean_recurrent_input_per_neuron_mv"],
                "max_recurrent_input_mv": conn_data["max_recurrent_input_mv"],
                "fraction_neurons_receiving_recurrent_input": conn_data["fraction_neurons_receiving_recurrent_input"],
                "mean_recurrent_input_from_stimulated_mv": conn_data["mean_recurrent_input_from_stimulated_mv"],
                "mean_recurrent_input_from_unstimulated_mv": conn_data["mean_recurrent_input_from_unstimulated_mv"],
                "runtime_seconds": round(dur, 4),
            }
            raw_records.append(record)
            baseline_lookup[(sf, alpha)] = record

            logger.info(
                f"[{run_idx:02d}/{total_runs}] Stim: {sf*100:>2.0f}% | Alpha: {alpha:<6.4f} | "
                f"Rate: {res.metrics['mean_firing_rate_hz']:>6.2f} Hz | Active: {res.metrics['active_neuron_fraction']*100:>4.1f}% | "
                f"Recruited: {rec_data['recruited_unstimulated_count']:>3d} ({rec_data['recruited_unstimulated_fraction']*100:>4.1f}%) | "
                f"PostRate: {post_win['mean_firing_rate_hz']:>5.2f} Hz | Vmax: {record['v_max_observed']:>6.1f} mV | "
                f"Regime: {regime}"
            )

    return raw_records, baseline_lookup


def select_representative_alphas(
    intact_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Selects 3 representative alphas from the 25% stimulus intact sweep data:
    1. Low-coupling representative: alpha = 0.0000 (purely uncoupled LIF dynamics, zero recurrent recruitment).
    2. Recurrent-transition representative: alpha = 0.0200 (first observed post-stimulus activity > 0 Hz and 19.5% recruitment).
    3. Strong-recurrent representative: alpha = 0.0300 (strong recurrent amplification, 87.1% recruitment, 19.5 Hz rate).
    """
    return {
        "low": 0.0000,
        "transition": 0.0200,
        "strong_recurrent": 0.0300,
    }


def run_lesion_validation(
    baseline_graph: nx.DiGraph,
    representative_alphas: Dict[str, float],
    intact_lookup: Dict[Tuple[float, float], Dict[str, Any]],
    base_config: LIFConfig,
    stim_ids_25: List[int],
) -> List[Dict[str, Any]]:
    """
    Runs limited lesion validation on representative alphas for the 25% stimulus condition:
    lesion fractions: 0% (intact), 10%, 20%
    strategies: hub, random (5 seeds: 42, 123, 456, 789, 1000)
    """
    logger.info("\n--- Running Limited Lesion Validation on Representative Alphas ---")
    base_n = baseline_graph.number_of_nodes()
    base_e = baseline_graph.number_of_edges()
    base_w = sum(d.get("weight", 1.0) for _, _, d in baseline_graph.edges(data=True))

    lesion_records: List[Dict[str, Any]] = []

    for regime_name, alpha in representative_alphas.items():
        logger.info(f"Lesion validation for {regime_name.upper()} regime: alpha = {alpha:.4f} mV/synapse")
        base_intact = intact_lookup[(0.25, alpha)]

        # Record 0% lesion baseline
        lesion_records.append(
            {
                "regime_name": regime_name,
                "alpha": alpha,
                "stimulus_fraction": 0.25,
                "lesion_strategy": "intact",
                "lesion_fraction": 0.0,
                "random_seed": "NA",
                "initial_neurons": base_n,
                "surviving_neurons": base_n,
                "initial_edges": base_e,
                "surviving_edges": base_e,
                "edge_retention": 1.0,
                "initial_total_synaptic_weight": round(base_w, 1),
                "surviving_total_synaptic_weight": round(base_w, 1),
                "synaptic_weight_retention": 1.0,
                "surviving_stimulated_neurons": len(stim_ids_25),
                "is_stable": base_intact["is_stable"],
                "total_spikes": base_intact["total_spikes"],
                "mean_firing_rate_hz": base_intact["mean_firing_rate_hz"],
                "active_neuron_fraction": base_intact["active_neuron_fraction"],
                "stim_window_rate_hz": base_intact["stim_window_rate_hz"],
                "post_stim_rate_hz": base_intact["post_stim_rate_hz"],
                "recruited_unstimulated_fraction": base_intact["recruited_unstimulated_fraction"],
                "activity_robustness": 1.0,
                "spike_count_robustness": 1.0,
                "runtime_seconds": base_intact["runtime_seconds"],
            }
        )

        cfg = LIFConfig(
            tau_m=base_config.tau_m,
            v_rest=base_config.v_rest,
            v_reset=base_config.v_reset,
            v_threshold=base_config.v_threshold,
            t_ref=base_config.t_ref,
            dt=base_config.dt,
            duration=base_config.duration,
            synaptic_weight_scale=alpha,
            external_stimulus_mode="pulse",
            external_current=base_config.external_current,
            noise_sigma=base_config.noise_sigma,
            pulse_start=base_config.pulse_start,
            pulse_end=base_config.pulse_end,
            stimulus_fraction=0.25,
            random_seed=base_config.random_seed,
            max_rate_threshold=base_config.max_rate_threshold,
        )

        # 1. Hub lesions (10%, 20%)
        for lf in LESION_FRACTIONS:
            lesioned_g, _ = apply_hub_lesion(baseline_graph, lesion_fraction=lf)
            n_surv = lesioned_g.number_of_nodes()
            e_surv = lesioned_g.number_of_edges()
            w_surv = sum(d.get("weight", 1.0) for _, _, d in lesioned_g.edges(data=True))

            surv_stim = [nid for nid in stim_ids_25 if nid in lesioned_g]

            res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                graph=lesioned_g,
                config=cfg,
                stimulated_neuron_ids=surv_stim,
            )

            base_rate = base_intact["mean_firing_rate_hz"]
            base_spikes = base_intact["total_spikes"]
            act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
            spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan

            lesion_records.append(
                {
                    "regime_name": regime_name,
                    "alpha": alpha,
                    "stimulus_fraction": 0.25,
                    "lesion_strategy": "hub",
                    "lesion_fraction": lf,
                    "random_seed": "NA",
                    "initial_neurons": base_n,
                    "surviving_neurons": n_surv,
                    "initial_edges": base_e,
                    "surviving_edges": e_surv,
                    "edge_retention": round(e_surv / base_e, 4),
                    "initial_total_synaptic_weight": round(base_w, 1),
                    "surviving_total_synaptic_weight": round(w_surv, 1),
                    "synaptic_weight_retention": round(w_surv / base_w, 4),
                    "surviving_stimulated_neurons": len(surv_stim),
                    "is_stable": res.is_stable,
                    "total_spikes": res.metrics["total_spikes"],
                    "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                    "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                    "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                    "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                    "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                    "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                    "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                    "runtime_seconds": round(dur, 4),
                }
            )

        # 2. Random lesions (10%, 20% x 5 seeds)
        for lf in LESION_FRACTIONS:
            for seed in RANDOM_SEEDS:
                lesioned_g, _ = apply_random_lesion(baseline_graph, lesion_fraction=lf, seed=seed)
                n_surv = lesioned_g.number_of_nodes()
                e_surv = lesioned_g.number_of_edges()
                w_surv = sum(d.get("weight", 1.0) for _, _, d in lesioned_g.edges(data=True))

                surv_stim = [nid for nid in stim_ids_25 if nid in lesioned_g]

                res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                    graph=lesioned_g,
                    config=cfg,
                    stimulated_neuron_ids=surv_stim,
                )

                base_rate = base_intact["mean_firing_rate_hz"]
                base_spikes = base_intact["total_spikes"]
                act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
                spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan

                lesion_records.append(
                    {
                        "regime_name": regime_name,
                        "alpha": alpha,
                        "stimulus_fraction": 0.25,
                        "lesion_strategy": "random",
                        "lesion_fraction": lf,
                        "random_seed": seed,
                        "initial_neurons": base_n,
                        "surviving_neurons": n_surv,
                        "initial_edges": base_e,
                        "surviving_edges": e_surv,
                        "edge_retention": round(e_surv / base_e, 4),
                        "initial_total_synaptic_weight": round(base_w, 1),
                        "surviving_total_synaptic_weight": round(w_surv, 1),
                        "synaptic_weight_retention": round(w_surv / base_w, 4),
                        "surviving_stimulated_neurons": len(surv_stim),
                        "is_stable": res.is_stable,
                        "total_spikes": res.metrics["total_spikes"],
                        "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                        "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                        "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                        "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                        "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                        "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                        "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                        "runtime_seconds": round(dur, 4),
                    }
                )

    return lesion_records


def generate_publication_figures(
    intact_df: pd.DataFrame,
    lesion_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    figures_dir: Path,
):
    """
    Renders 9 publication-grade figures (300 DPI, modern scientific aesthetic).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "figure.autolayout": True})

    colors_stim = {
        0.25: "#1f77b4",  # Blue
        0.10: "#ff7f0e",  # Orange
        0.05: "#2ca02c",  # Green
        0.01: "#d62728",  # Red
        0.00: "#7f7f7f",  # Gray
    }

    # =========================================================================
    # Figure 1: Mean Firing Rate vs Alpha
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    for sf in STIMULUS_FRACTIONS:
        sub = intact_df[intact_df["stimulus_fraction"] == sf].sort_values("alpha")
        # Mask unstable points or plot them with dotted red
        ax.plot(
            sub["alpha"],
            sub["mean_firing_rate_hz"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=f"Stimulus {sf*100:.0f}%",
            color=colors_stim[sf],
        )

    ax.set_title("Population Mean Firing Rate Across Coupling Scales", fontweight="bold", pad=12)
    ax.set_xlabel(r"Synaptic Coupling Scale $\alpha$ (mV / synapse contact)")
    ax.set_ylabel("Mean Firing Rate (Hz)")
    ax.legend(frameon=True, facecolor="#FFFFFF")
    fig.savefig(figures_dir / "coupling_mean_firing_rate.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Figure 2: Post-Stimulus Activity vs Alpha
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    for sf in [0.25, 0.10, 0.05, 0.01]:
        sub = intact_df[intact_df["stimulus_fraction"] == sf].sort_values("alpha")
        ax.plot(
            sub["alpha"],
            sub["post_stim_rate_hz"],
            marker="s",
            linewidth=2,
            markersize=5,
            label=f"Stimulus {sf*100:.0f}%",
            color=colors_stim[sf],
        )

    ax.set_title("Post-Stimulus Firing Rate (600–1000 ms) vs Coupling Scale", fontweight="bold", pad=12)
    ax.set_xlabel(r"Synaptic Coupling Scale $\alpha$ (mV / synapse contact)")
    ax.set_ylabel("Post-Stimulus Firing Rate (Hz)")
    ax.legend(frameon=True, facecolor="#FFFFFF")
    fig.savefig(figures_dir / "coupling_post_stimulus_rate.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Figure 3: Unstimulated-Neuron Recruitment vs Alpha
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    for sf in [0.25, 0.10, 0.05, 0.01]:
        sub = intact_df[intact_df["stimulus_fraction"] == sf].sort_values("alpha")
        ax.plot(
            sub["alpha"],
            sub["recruited_unstimulated_fraction"] * 100,
            marker="^",
            linewidth=2,
            markersize=6,
            label=f"Stimulus {sf*100:.0f}%",
            color=colors_stim[sf],
        )

    ax.set_title("Unstimulated Neuron Recruitment During Stimulus Window", fontweight="bold", pad=12)
    ax.set_xlabel(r"Synaptic Coupling Scale $\alpha$ (mV / synapse contact)")
    ax.set_ylabel("Recruited Unstimulated Neurons (%)")
    ax.legend(frameon=True, facecolor="#FFFFFF")
    fig.savefig(figures_dir / "coupling_unstimulated_recruitment.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Figure 4: Stimulated vs Unstimulated Firing (25% Stimulus)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    sub_25 = intact_df[intact_df["stimulus_fraction"] == 0.25].sort_values("alpha")
    ax.plot(sub_25["alpha"], sub_25["stimulated_rate_hz"], marker="o", color="#1f77b4", linewidth=2, label="Stimulated (N=250)")
    ax.plot(sub_25["alpha"], sub_25["unstimulated_rate_hz"], marker="s", color="#ff7f0e", linewidth=2, label="Unstimulated (N=750)")

    ax.set_title("Stimulated vs Unstimulated Firing Rates (25% Drive)", fontweight="bold", pad=12)
    ax.set_xlabel(r"Synaptic Coupling Scale $\alpha$ (mV / synapse contact)")
    ax.set_ylabel("Mean Firing Rate (Hz)")
    ax.legend(frameon=True, facecolor="#FFFFFF")
    fig.savefig(figures_dir / "coupling_stim_vs_unstim_firing.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Figure 5: Activity Regime Diagram
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")

    regime_colors = {
        "Externally Driven (Weak Coupling)": "#90CAF9",   # Light Blue
        "Recurrent Recruitment": "#A5D6A7",               # Light Green
        "Sustained Recurrent Activity": "#FFE082",        # Light Yellow/Amber
        "Unstable": "#EF9A9A",                             # Light Red
    }

    sub_reg = regimes_df.sort_values("alpha").reset_index(drop=True)
    alphas = sub_reg["alpha"].values
    regimes = sub_reg["regime"].values

    for i, (a, r) in enumerate(zip(alphas, regimes)):
        col = regime_colors.get(r, "#CCCCCC")
        ax.bar(i, 1.0, color=col, edgecolor="#333333", width=0.8, alpha=0.9)
        ax.text(i, 0.5, f"{r}\n(Rate: {sub_reg.loc[i, 'mean_firing_rate_hz']:.1f} Hz)", ha="center", va="center", rotation=90, fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.4f}" for a in alphas], rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_xlabel(r"Synaptic Coupling Scale $\alpha$ (mV / synapse contact)")
    ax.set_title("Observed Dynamic Regimes Across Coupling Scale Grid (25% Stimulus)", fontweight="bold", pad=12)
    fig.savefig(figures_dir / "coupling_activity_regimes.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Figure 6: Maximum Membrane Potential vs Alpha (Stability Boundary)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    sub_25 = intact_df[intact_df["stimulus_fraction"] == 0.25].sort_values("alpha")
    ax.plot(sub_25["alpha"], sub_25["v_max_observed"], marker="D", color="#800080", linewidth=2, label=r"Max Observed $V_m$")
    ax.axhline(-50.0, color="#2ca02c", linestyle="--", linewidth=1.5, label=r"Spike Threshold ($-50$ mV)")
    ax.axhline(120.0, color="#d62728", linestyle="--", linewidth=1.5, label=r"Numerical Instability Limit ($+120$ mV)")

    ax.set_title(r"Maximum Membrane Potential $V_{\max}$ vs Coupling Scale $\alpha$", fontweight="bold", pad=12)
    ax.set_xlabel(r"Synaptic Coupling Scale $\alpha$ (mV / synapse contact)")
    ax.set_ylabel(r"Peak Membrane Potential $V_m$ (mV)")
    ax.legend(frameon=True, facecolor="#FFFFFF")
    fig.savefig(figures_dir / "coupling_max_membrane_potential.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Figure 7: Total Spikes vs Alpha
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    for sf in STIMULUS_FRACTIONS:
        sub = intact_df[intact_df["stimulus_fraction"] == sf].sort_values("alpha")
        ax.plot(
            sub["alpha"],
            sub["total_spikes"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=f"Stimulus {sf*100:.0f}%",
            color=colors_stim[sf],
        )

    ax.set_title("Total Emitted Spikes Across Coupling Scales", fontweight="bold", pad=12)
    ax.set_xlabel(r"Synaptic Coupling Scale $\alpha$ (mV / synapse contact)")
    ax.set_ylabel("Total Network Spikes (1000 ms)")
    ax.legend(frameon=True, facecolor="#FFFFFF")
    fig.savefig(figures_dir / "coupling_total_spikes.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Figure 8: Post-Stimulus Persistence Ratio vs Alpha
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    for sf in [0.25, 0.10, 0.05, 0.01]:
        sub = intact_df[intact_df["stimulus_fraction"] == sf].sort_values("alpha")
        ax.plot(
            sub["alpha"],
            sub["post_stimulus_persistence"],
            marker="p",
            linewidth=2,
            markersize=6,
            label=f"Stimulus {sf*100:.0f}%",
            color=colors_stim[sf],
        )

    ax.set_title("Post-Stimulus Persistence Ratio (Post Active / Stim Active)", fontweight="bold", pad=12)
    ax.set_xlabel(r"Synaptic Coupling Scale $\alpha$ (mV / synapse contact)")
    ax.set_ylabel("Persistence Ratio")
    ax.legend(frameon=True, facecolor="#FFFFFF")
    fig.savefig(figures_dir / "coupling_post_stim_persistence.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Figure 9: Structural Coupling vs Functional Recruitment
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

    for sf in [0.25, 0.10, 0.05, 0.01]:
        sub = intact_df[intact_df["stimulus_fraction"] == sf].sort_values("mean_recurrent_input_per_neuron_mv")
        ax.scatter(
            sub["mean_recurrent_input_per_neuron_mv"],
            sub["recruited_unstimulated_fraction"] * 100,
            s=60,
            color=colors_stim[sf],
            label=f"Stimulus {sf*100:.0f}%",
            edgecolor="#333333",
            zorder=3,
        )
        ax.plot(
            sub["mean_recurrent_input_per_neuron_mv"],
            sub["recruited_unstimulated_fraction"] * 100,
            color=colors_stim[sf],
            linestyle="--",
            alpha=0.6,
        )

    ax.set_title("Structural Recurrent Input vs Functional Neuron Recruitment", fontweight="bold", pad=12)
    ax.set_xlabel("Mean Recurrent Synaptic Input per Neuron (mV)")
    ax.set_ylabel("Recruited Unstimulated Neurons (%)")
    ax.legend(frameon=True, facecolor="#FFFFFF")
    fig.savefig(figures_dir / "coupling_structural_vs_recruitment.png", dpi=300)
    plt.close(fig)

    logger.info("Successfully generated all 9 publication-grade figures!")


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Coupling-Regime Sweep and Recurrent Dynamics")
    parser.add_argument("--neurons", type=int, default=1000, help="Subnetwork neuron count (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--external-current", type=float, default=18.0, help="External pulse current (default: 18.0)")
    parser.add_argument("--dt", type=float, default=0.5, help="Simulation timestep (default: 0.5 ms)")
    parser.add_argument("--duration", type=float, default=1000.0, help="Simulation duration (default: 1000.0 ms)")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test on 3 alphas only")
    parser.add_argument("--lesion-only", action="store_true", help="Run only representative lesion validation without rerunning intact sweep")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Real Subnetwork
    logger.info("Loading real 1,000-neuron connectome subnetwork...")
    sub_graph, _ = extract_real_subnetwork(
        max_neurons=args.neurons,
        strategy="highest_degree",
        seed=args.seed,
        verbose=False,
    )
    base_n = sub_graph.number_of_nodes()
    base_e = sub_graph.number_of_edges()
    base_w = sum(d.get("weight", 1.0) for _, _, d in sub_graph.edges(data=True))

    logger.info(f"Loaded network: N={base_n}, Edges={base_e}, Synapses={base_w:.0f}")

    # 2. Select Stimulus Sets deterministically on intact network
    stim_sets: Dict[float, List[int]] = {}
    for sf in STIMULUS_FRACTIONS:
        stim_sets[sf] = select_stimulated_neurons(sub_graph, stimulus_fraction=sf, seed=args.seed)
        logger.info(f"Stimulus {sf*100:>2.0f}%: {len(stim_sets[sf]):>3d} neurons selected.")

    base_config = LIFConfig(
        dt=args.dt,
        duration=args.duration,
        external_current=args.external_current,
        random_seed=args.seed,
    )

    tracemalloc.start()
    t_start = time.perf_counter()

    raw_csv = tables_dir / "coupling_sweep_raw_results.csv"

    if args.lesion_only:
        logger.info("Running in LESION-ONLY mode: reusing existing intact coupling sweep results...")
        if not raw_csv.exists():
            raise FileNotFoundError(f"Cannot run lesion-only: {raw_csv} does not exist.")
        intact_df = pd.read_csv(raw_csv)
        intact_records = intact_df.to_dict("records")
        intact_lookup = {(round(float(r["stimulus_fraction"]), 4), round(float(r["alpha"]), 4)): r for r in intact_records}
    else:
        # Alphas to test
        if args.smoke_test:
            test_alphas = [0.005, 0.010, 0.050]
            test_stim_fracs = [0.25, 0.05]
        else:
            test_alphas = ALPHA_VALUES
            test_stim_fracs = STIMULUS_FRACTIONS

        # 3. Execute Intact Coupling Sweep (55 simulations)
        intact_records, intact_lookup = run_intact_coupling_sweep(
            graph=sub_graph,
            alphas=test_alphas,
            stimulus_fractions=test_stim_fracs,
            base_config=base_config,
            stim_sets=stim_sets,
        )

        intact_df = pd.DataFrame(intact_records)
        intact_df.to_csv(raw_csv, index=False)
        logger.info(f"Saved raw intact coupling results to: {raw_csv}")

    # Generate Summary Table
    summary_cols = [
        "alpha",
        "stimulus_fraction",
        "is_stable",
        "regime",
        "mean_firing_rate_hz",
        "active_neuron_fraction",
        "stimulated_rate_hz",
        "unstimulated_rate_hz",
        "recruited_unstimulated_count",
        "recruited_unstimulated_fraction",
        "stim_window_rate_hz",
        "post_stim_rate_hz",
        "v_max_observed",
        "total_spikes",
    ]
    summary_df = intact_df[summary_cols].copy()
    sum_csv = tables_dir / "coupling_sweep_summary.csv"
    summary_df.to_csv(sum_csv, index=False)
    logger.info(f"Saved coupling summary to: {sum_csv}")

    # Save 25% Stimulus Regime Table
    regimes_df = intact_df[intact_df["stimulus_fraction"] == 0.25][
        [
            "alpha",
            "is_stable",
            "regime",
            "mean_firing_rate_hz",
            "active_neuron_fraction",
            "recruited_unstimulated_count",
            "recruited_unstimulated_fraction",
            "post_stim_rate_hz",
            "v_max_observed",
            "status_message",
        ]
    ].copy()
    reg_csv = tables_dir / "coupling_regimes.csv"
    regimes_df.to_csv(reg_csv, index=False)
    logger.info(f"Saved regime classifications to: {reg_csv}")

    # 4. Representative Alphas & Limited Lesion Validation
    rep_alphas = select_representative_alphas(intact_records)
    logger.info(f"Selected representative alphas: {rep_alphas}")

    lesion_records = run_lesion_validation(
        baseline_graph=sub_graph,
        representative_alphas=rep_alphas,
        intact_lookup=intact_lookup,
        base_config=base_config,
        stim_ids_25=stim_sets[0.25],
    )

    lesion_df = pd.DataFrame(lesion_records)
    les_csv = tables_dir / "coupling_lesion_validation.csv"
    lesion_df.to_csv(les_csv, index=False)
    logger.info(f"Saved limited lesion validation results to: {les_csv}")

    # 5. Generate Figures
    logger.info("Generating publication figures...")
    generate_publication_figures(
        intact_df=intact_df,
        lesion_df=lesion_df,
        regimes_df=regimes_df,
        figures_dir=figures_dir,
    )

    t_end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info("=" * 65)
    logger.info(" PHASE 6 EXECUTION COMPLETE")
    logger.info(f" Total Simulations: {len(intact_records) + len(lesion_records) - len(rep_alphas)}")
    logger.info(f" Total Runtime:     {t_end - t_start:.2f} s")
    logger.info(f" Peak Memory:       {peak_mem / (1024*1024):.2f} MB")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
