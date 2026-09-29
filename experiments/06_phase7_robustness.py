"""
Phase 7: Robustness & Sensitivity Validation
Drosophila Hemibrain Connectome-Constrained LIF Network

Sub-experiments:
- 7A: Local alpha sensitivity around the transition (alpha in [0.0150, 0.0175, 0.0200, 0.0225, 0.0250])
- 7B: Stimulus-fraction sensitivity (stimulus_fraction in [0.05, 0.10, 0.25] for alpha in [0.0200, 0.0300])
- 7C: Lesion-fraction sensitivity (lesion_fraction in [0.05, 0.10, 0.20] for alpha in [0.0200, 0.0300])
- 7D: Numerical timestep sensitivity (dt in [0.25, 0.50, 1.00] ms for alpha in [0.0200, 0.0300])
- 7E: Subnetwork selection sensitivity (highest_degree vs random seed 2026 for alpha in [0.0200, 0.0300])

Simulation accounting:
- 7A: 5 simulations
- 7B: 78 simulations (2 alphas x 3 stimulus fractions x [1 intact + 2 hub + 10 random])
- 7C: 38 simulations (2 alphas x [1 intact + 3 hub + 15 random])
- 7D: 24 simulations (2 alphas x 3 dt x [1 intact + 2 hub + 1 random])
- 7E: 12 simulations (2 networks x 2 alphas x [1 intact + 1 hub 20% + 1 random 20%])
Total simulations: exactly 157 simulations.
"""

import argparse
import importlib.util
import logging
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

# Import modules from src
from src.data_loader import extract_real_subnetwork
from src.graph_builder import get_graph_summary
from src.lesion import apply_hub_lesion, apply_random_lesion
from src.simulation import LIFConfig, LIFNetwork, SimulationResult, select_stimulated_neurons

# Dynamically import helper metrics from Phase 6
sweep_path = Path(__file__).resolve().parent / "05_coupling_sweep.py"
spec = importlib.util.spec_from_file_location("exp05", sweep_path)
exp05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp05)

compute_activity_windows = exp05.compute_activity_windows
compute_recurrent_activity_metrics = exp05.compute_recurrent_activity_metrics
compute_connectivity_metrics = exp05.compute_connectivity_metrics
classify_coupling_regime = exp05.classify_coupling_regime
run_single_sweep_simulation = exp05.run_single_sweep_simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase7_robustness")

# Constants
RANDOM_SEEDS = [42, 123, 456, 789, 1000]
BASE_ALPHA_VALUES = [0.0200, 0.0300]


def run_experiment_7a(
    sub_graph: nx.DiGraph,
    stim_ids_25: List[int],
    base_config: LIFConfig,
) -> List[Dict[str, Any]]:
    """
    Experiment 7A: Local alpha sensitivity around the transition.
    Alphas: 0.0150, 0.0175, 0.0200, 0.0225, 0.0250 (5 simulations).
    Intact network, 25% stimulus.
    """
    logger.info("\n" + "=" * 65)
    logger.info(" EXPERIMENT 7A: Local Alpha Sensitivity Around Transition")
    logger.info("=" * 65)
    alphas = [0.0150, 0.0175, 0.0200, 0.0225, 0.0250]
    records = []

    base_n = sub_graph.number_of_nodes()
    base_e = sub_graph.number_of_edges()
    base_w = sum(d.get("weight", 1.0) for _, _, d in sub_graph.edges(data=True))

    for alpha in alphas:
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

        res, win_data, rec_data, _, dur = run_single_sweep_simulation(
            graph=sub_graph,
            config=cfg,
            stimulated_neuron_ids=stim_ids_25,
        )

        regime = classify_coupling_regime(
            is_stable=res.is_stable,
            recruited_unstimulated_fraction=rec_data["recruited_unstimulated_fraction"],
            post_stimulus_rate_hz=win_data["post_stimulus"]["mean_firing_rate_hz"],
        )

        rec = {
            "experiment": "alpha_sensitivity",
            "alpha": alpha,
            "stimulus_fraction": 0.25,
            "lesion_fraction": 0.0,
            "lesion_strategy": "intact",
            "random_seed": "NA",
            "dt": base_config.dt,
            "network_selection": "highest_degree",
            "network_seed": 42,
            "initial_neurons": base_n,
            "surviving_neurons": base_n,
            "initial_edges": base_e,
            "surviving_edges": base_e,
            "edge_retention": 1.0,
            "initial_total_synaptic_weight": round(base_w, 1),
            "surviving_total_synaptic_weight": round(base_w, 1),
            "synaptic_weight_retention": 1.0,
            "surviving_stimulated_neurons": len(stim_ids_25),
            "is_stable": res.is_stable,
            "v_max_observed": res.metrics.get("v_max_observed", np.nan),
            "total_spikes": res.metrics["total_spikes"],
            "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
            "active_neuron_fraction": res.metrics["active_neuron_fraction"],
            "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
            "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
            "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
            "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
            "persistence_ratio": rec_data["post_stimulus_persistence"],
            "activity_robustness": 1.0,
            "spike_count_robustness": 1.0,
            "regime": regime,
            "runtime_seconds": round(dur, 4),
        }
        records.append(rec)
        logger.info(
            f"Alpha: {alpha:.4f} | Rate: {rec['mean_firing_rate_hz']:>6.2f} Hz | "
            f"Recruited: {rec['recruited_unstimulated_count']:>3d} ({rec['recruited_unstimulated_fraction']*100:>4.1f}%) | "
            f"PostRate: {rec['post_stim_rate_hz']:>5.2f} Hz | Vmax: {rec['v_max_observed']:>6.1f} mV | {regime}"
        )

    return records


def run_experiment_7b(
    sub_graph: nx.DiGraph,
    stim_sets: Dict[float, List[int]],
    base_config: LIFConfig,
) -> List[Dict[str, Any]]:
    """
    Experiment 7B: Stimulus-fraction sensitivity across [0.05, 0.10, 0.25]
    for alpha in [0.0200, 0.0300].
    Conditions per (alpha, sf): 1 intact + 2 hub (10%, 20%) + 10 random (10%, 20% x 5 seeds).
    Total runs: 2 alphas x 3 stimulus fractions x 13 = 78 simulations.
    """
    logger.info("\n" + "=" * 65)
    logger.info(" EXPERIMENT 7B: Stimulus-Fraction Sensitivity")
    logger.info("=" * 65)
    stim_fractions = [0.05, 0.10, 0.25]
    alphas = BASE_ALPHA_VALUES
    records = []

    base_n = sub_graph.number_of_nodes()
    base_e = sub_graph.number_of_edges()
    base_w = sum(d.get("weight", 1.0) for _, _, d in sub_graph.edges(data=True))

    for alpha in alphas:
        for sf in stim_fractions:
            stim_ids = stim_sets[sf]
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

            # 1. Intact baseline
            res_intact, win_intact, rec_intact, _, dur_intact = run_single_sweep_simulation(
                graph=sub_graph,
                config=cfg,
                stimulated_neuron_ids=stim_ids,
            )
            base_rate = res_intact.metrics["mean_firing_rate_hz"]
            base_spikes = res_intact.metrics["total_spikes"]

            intact_rec = {
                "experiment": "stimulus_sensitivity",
                "alpha": alpha,
                "stimulus_fraction": sf,
                "lesion_fraction": 0.0,
                "lesion_strategy": "intact",
                "random_seed": "NA",
                "dt": base_config.dt,
                "network_selection": "highest_degree",
                "network_seed": 42,
                "initial_neurons": base_n,
                "surviving_neurons": base_n,
                "initial_edges": base_e,
                "surviving_edges": base_e,
                "edge_retention": 1.0,
                "initial_total_synaptic_weight": round(base_w, 1),
                "surviving_total_synaptic_weight": round(base_w, 1),
                "synaptic_weight_retention": 1.0,
                "surviving_stimulated_neurons": len(stim_ids),
                "is_stable": res_intact.is_stable,
                "v_max_observed": res_intact.metrics.get("v_max_observed", np.nan),
                "total_spikes": res_intact.metrics["total_spikes"],
                "mean_firing_rate_hz": res_intact.metrics["mean_firing_rate_hz"],
                "active_neuron_fraction": res_intact.metrics["active_neuron_fraction"],
                "stim_window_rate_hz": win_intact["stimulus"]["mean_firing_rate_hz"],
                "post_stim_rate_hz": win_intact["post_stimulus"]["mean_firing_rate_hz"],
                "recruited_unstimulated_count": rec_intact["recruited_unstimulated_count"],
                "recruited_unstimulated_fraction": rec_intact["recruited_unstimulated_fraction"],
                "persistence_ratio": rec_intact["post_stimulus_persistence"],
                "activity_robustness": 1.0,
                "spike_count_robustness": 1.0,
                "runtime_seconds": round(dur_intact, 4),
            }
            records.append(intact_rec)

            # 2. Hub lesions (10%, 20%)
            for lf in [0.10, 0.20]:
                lesioned_g, _ = apply_hub_lesion(sub_graph, lesion_fraction=lf)
                n_surv = lesioned_g.number_of_nodes()
                e_surv = lesioned_g.number_of_edges()
                w_surv = sum(d.get("weight", 1.0) for _, _, d in lesioned_g.edges(data=True))
                surv_stim = [nid for nid in stim_ids if nid in lesioned_g]

                res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                    graph=lesioned_g,
                    config=cfg,
                    stimulated_neuron_ids=surv_stim,
                )

                act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
                spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan

                records.append(
                    {
                        "experiment": "stimulus_sensitivity",
                        "alpha": alpha,
                        "stimulus_fraction": sf,
                        "lesion_fraction": lf,
                        "lesion_strategy": "hub",
                        "random_seed": "NA",
                        "dt": base_config.dt,
                        "network_selection": "highest_degree",
                        "network_seed": 42,
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
                        "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                        "total_spikes": res.metrics["total_spikes"],
                        "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                        "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                        "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                        "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                        "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                        "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                        "persistence_ratio": rec_data["post_stimulus_persistence"],
                        "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                        "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                        "runtime_seconds": round(dur, 4),
                    }
                )

            # 3. Random lesions (10%, 20% x 5 seeds)
            for lf in [0.10, 0.20]:
                for seed in RANDOM_SEEDS:
                    lesioned_g, _ = apply_random_lesion(sub_graph, lesion_fraction=lf, seed=seed)
                    n_surv = lesioned_g.number_of_nodes()
                    e_surv = lesioned_g.number_of_edges()
                    w_surv = sum(d.get("weight", 1.0) for _, _, d in lesioned_g.edges(data=True))
                    surv_stim = [nid for nid in stim_ids if nid in lesioned_g]

                    res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                        graph=lesioned_g,
                        config=cfg,
                        stimulated_neuron_ids=surv_stim,
                    )

                    act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
                    spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan

                    records.append(
                        {
                            "experiment": "stimulus_sensitivity",
                            "alpha": alpha,
                            "stimulus_fraction": sf,
                            "lesion_fraction": lf,
                            "lesion_strategy": "random",
                            "random_seed": seed,
                            "dt": base_config.dt,
                            "network_selection": "highest_degree",
                            "network_seed": 42,
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
                            "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                            "total_spikes": res.metrics["total_spikes"],
                            "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                            "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                            "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                            "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                            "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                            "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                            "persistence_ratio": rec_data["post_stimulus_persistence"],
                            "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                            "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                            "runtime_seconds": round(dur, 4),
                        }
                    )

            logger.info(f"Completed Stimulus Sensitivity for alpha={alpha:.4f}, sf={sf*100:.0f}% (13 simulations)")

    return records


def run_experiment_7c(
    sub_graph: nx.DiGraph,
    stim_ids_25: List[int],
    base_config: LIFConfig,
) -> List[Dict[str, Any]]:
    """
    Experiment 7C: Lesion-fraction sensitivity across [0.05, 0.10, 0.20]
    for alpha in [0.0200, 0.0300] at 25% stimulus.
    Per alpha: 1 intact + 3 hub + 15 random (3 lf x 5 seeds).
    Total runs: 2 alphas x 19 = 38 simulations.
    """
    logger.info("\n" + "=" * 65)
    logger.info(" EXPERIMENT 7C: Lesion-Fraction Sensitivity")
    logger.info("=" * 65)
    lesion_fractions = [0.05, 0.10, 0.20]
    alphas = BASE_ALPHA_VALUES
    records = []

    base_n = sub_graph.number_of_nodes()
    base_e = sub_graph.number_of_edges()
    base_w = sum(d.get("weight", 1.0) for _, _, d in sub_graph.edges(data=True))

    for alpha in alphas:
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

        # 1. Intact baseline
        res_intact, win_intact, rec_intact, _, dur_intact = run_single_sweep_simulation(
            graph=sub_graph,
            config=cfg,
            stimulated_neuron_ids=stim_ids_25,
        )
        base_rate = res_intact.metrics["mean_firing_rate_hz"]
        base_spikes = res_intact.metrics["total_spikes"]

        intact_rec = {
            "experiment": "lesion_sensitivity",
            "alpha": alpha,
            "stimulus_fraction": 0.25,
            "lesion_fraction": 0.0,
            "lesion_strategy": "intact",
            "random_seed": "NA",
            "dt": base_config.dt,
            "network_selection": "highest_degree",
            "network_seed": 42,
            "initial_neurons": base_n,
            "surviving_neurons": base_n,
            "initial_edges": base_e,
            "surviving_edges": base_e,
            "edge_retention": 1.0,
            "initial_total_synaptic_weight": round(base_w, 1),
            "surviving_total_synaptic_weight": round(base_w, 1),
            "synaptic_weight_retention": 1.0,
            "surviving_stimulated_neurons": len(stim_ids_25),
            "is_stable": res_intact.is_stable,
            "v_max_observed": res_intact.metrics.get("v_max_observed", np.nan),
            "total_spikes": res_intact.metrics["total_spikes"],
            "mean_firing_rate_hz": res_intact.metrics["mean_firing_rate_hz"],
            "active_neuron_fraction": res_intact.metrics["active_neuron_fraction"],
            "stim_window_rate_hz": win_intact["stimulus"]["mean_firing_rate_hz"],
            "post_stim_rate_hz": win_intact["post_stimulus"]["mean_firing_rate_hz"],
            "recruited_unstimulated_count": rec_intact["recruited_unstimulated_count"],
            "recruited_unstimulated_fraction": rec_intact["recruited_unstimulated_fraction"],
            "persistence_ratio": rec_intact["post_stimulus_persistence"],
            "activity_robustness": 1.0,
            "spike_count_robustness": 1.0,
            "runtime_seconds": round(dur_intact, 4),
        }
        records.append(intact_rec)

        # 2. Hub lesions (5%, 10%, 20%)
        for lf in lesion_fractions:
            lesioned_g, _ = apply_hub_lesion(sub_graph, lesion_fraction=lf)
            n_surv = lesioned_g.number_of_nodes()
            e_surv = lesioned_g.number_of_edges()
            w_surv = sum(d.get("weight", 1.0) for _, _, d in lesioned_g.edges(data=True))
            surv_stim = [nid for nid in stim_ids_25 if nid in lesioned_g]

            res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                graph=lesioned_g,
                config=cfg,
                stimulated_neuron_ids=surv_stim,
            )

            act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
            spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan

            records.append(
                {
                    "experiment": "lesion_sensitivity",
                    "alpha": alpha,
                    "stimulus_fraction": 0.25,
                    "lesion_fraction": lf,
                    "lesion_strategy": "hub",
                    "random_seed": "NA",
                    "dt": base_config.dt,
                    "network_selection": "highest_degree",
                    "network_seed": 42,
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
                    "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                    "total_spikes": res.metrics["total_spikes"],
                    "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                    "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                    "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                    "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                    "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                    "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                    "persistence_ratio": rec_data["post_stimulus_persistence"],
                    "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                    "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                    "runtime_seconds": round(dur, 4),
                }
            )

        # 3. Random lesions (5%, 10%, 20% x 5 seeds)
        for lf in lesion_fractions:
            for seed in RANDOM_SEEDS:
                lesioned_g, _ = apply_random_lesion(sub_graph, lesion_fraction=lf, seed=seed)
                n_surv = lesioned_g.number_of_nodes()
                e_surv = lesioned_g.number_of_edges()
                w_surv = sum(d.get("weight", 1.0) for _, _, d in lesioned_g.edges(data=True))
                surv_stim = [nid for nid in stim_ids_25 if nid in lesioned_g]

                res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                    graph=lesioned_g,
                    config=cfg,
                    stimulated_neuron_ids=surv_stim,
                )

                act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
                spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan

                records.append(
                    {
                        "experiment": "lesion_sensitivity",
                        "alpha": alpha,
                        "stimulus_fraction": 0.25,
                        "lesion_fraction": lf,
                        "lesion_strategy": "random",
                        "random_seed": seed,
                        "dt": base_config.dt,
                        "network_selection": "highest_degree",
                        "network_seed": 42,
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
                        "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                        "total_spikes": res.metrics["total_spikes"],
                        "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                        "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                        "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                        "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                        "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                        "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                        "persistence_ratio": rec_data["post_stimulus_persistence"],
                        "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                        "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                        "runtime_seconds": round(dur, 4),
                    }
                )

        logger.info(f"Completed Lesion Sensitivity for alpha={alpha:.4f} (19 simulations)")

    return records


def run_experiment_7d(
    sub_graph: nx.DiGraph,
    stim_ids_25: List[int],
    base_config: LIFConfig,
) -> List[Dict[str, Any]]:
    """
    Experiment 7D: Numerical timestep sensitivity across dt in [0.25, 0.50, 1.00] ms
    for alpha in [0.0200, 0.0300] at 25% stimulus (fixed physical duration 1000 ms).
    Conditions per (alpha, dt): intact, hub 10%, hub 20%, random 20% (seed 42).
    Total runs: 2 alphas x 3 dt x 4 conditions = 24 simulations.
    """
    logger.info("\n" + "=" * 65)
    logger.info(" EXPERIMENT 7D: Numerical Timestep Sensitivity")
    logger.info("=" * 65)
    dts = [0.25, 0.50, 1.00]
    alphas = BASE_ALPHA_VALUES
    records = []

    base_n = sub_graph.number_of_nodes()
    base_e = sub_graph.number_of_edges()
    base_w = sum(d.get("weight", 1.0) for _, _, d in sub_graph.edges(data=True))

    for alpha in alphas:
        for dt_val in dts:
            cfg = LIFConfig(
                tau_m=base_config.tau_m,
                v_rest=base_config.v_rest,
                v_reset=base_config.v_reset,
                v_threshold=base_config.v_threshold,
                t_ref=base_config.t_ref,
                dt=dt_val,
                duration=1000.0,
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

            # 1. Intact baseline for this (alpha, dt)
            res_intact, win_intact, rec_intact, _, dur_intact = run_single_sweep_simulation(
                graph=sub_graph,
                config=cfg,
                stimulated_neuron_ids=stim_ids_25,
            )
            base_rate = res_intact.metrics["mean_firing_rate_hz"]
            base_spikes = res_intact.metrics["total_spikes"]

            intact_rec = {
                "experiment": "timestep_sensitivity",
                "alpha": alpha,
                "stimulus_fraction": 0.25,
                "lesion_fraction": 0.0,
                "lesion_strategy": "intact",
                "random_seed": "NA",
                "dt": dt_val,
                "network_selection": "highest_degree",
                "network_seed": 42,
                "initial_neurons": base_n,
                "surviving_neurons": base_n,
                "initial_edges": base_e,
                "surviving_edges": base_e,
                "edge_retention": 1.0,
                "initial_total_synaptic_weight": round(base_w, 1),
                "surviving_total_synaptic_weight": round(base_w, 1),
                "synaptic_weight_retention": 1.0,
                "surviving_stimulated_neurons": len(stim_ids_25),
                "is_stable": res_intact.is_stable,
                "v_max_observed": res_intact.metrics.get("v_max_observed", np.nan),
                "total_spikes": res_intact.metrics["total_spikes"],
                "mean_firing_rate_hz": res_intact.metrics["mean_firing_rate_hz"],
                "active_neuron_fraction": res_intact.metrics["active_neuron_fraction"],
                "stim_window_rate_hz": win_intact["stimulus"]["mean_firing_rate_hz"],
                "post_stim_rate_hz": win_intact["post_stimulus"]["mean_firing_rate_hz"],
                "recruited_unstimulated_count": rec_intact["recruited_unstimulated_count"],
                "recruited_unstimulated_fraction": rec_intact["recruited_unstimulated_fraction"],
                "persistence_ratio": rec_intact["post_stimulus_persistence"],
                "activity_robustness": 1.0,
                "spike_count_robustness": 1.0,
                "runtime_seconds": round(dur_intact, 4),
            }
            records.append(intact_rec)

            # 2. Hub 10%
            les_g_h10, _ = apply_hub_lesion(sub_graph, lesion_fraction=0.10)
            surv_stim = [nid for nid in stim_ids_25 if nid in les_g_h10]
            res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                graph=les_g_h10, config=cfg, stimulated_neuron_ids=surv_stim
            )
            act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
            spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan
            records.append(
                {
                    "experiment": "timestep_sensitivity",
                    "alpha": alpha,
                    "stimulus_fraction": 0.25,
                    "lesion_fraction": 0.10,
                    "lesion_strategy": "hub",
                    "random_seed": "NA",
                    "dt": dt_val,
                    "network_selection": "highest_degree",
                    "network_seed": 42,
                    "initial_neurons": base_n,
                    "surviving_neurons": les_g_h10.number_of_nodes(),
                    "initial_edges": base_e,
                    "surviving_edges": les_g_h10.number_of_edges(),
                    "edge_retention": round(les_g_h10.number_of_edges() / base_e, 4),
                    "initial_total_synaptic_weight": round(base_w, 1),
                    "surviving_total_synaptic_weight": round(sum(d.get("weight", 1.0) for _, _, d in les_g_h10.edges(data=True)), 1),
                    "synaptic_weight_retention": round(sum(d.get("weight", 1.0) for _, _, d in les_g_h10.edges(data=True)) / base_w, 4),
                    "surviving_stimulated_neurons": len(surv_stim),
                    "is_stable": res.is_stable,
                    "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                    "total_spikes": res.metrics["total_spikes"],
                    "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                    "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                    "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                    "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                    "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                    "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                    "persistence_ratio": rec_data["post_stimulus_persistence"],
                    "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                    "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                    "runtime_seconds": round(dur, 4),
                }
            )

            # 3. Hub 20%
            les_g_h20, _ = apply_hub_lesion(sub_graph, lesion_fraction=0.20)
            surv_stim = [nid for nid in stim_ids_25 if nid in les_g_h20]
            res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                graph=les_g_h20, config=cfg, stimulated_neuron_ids=surv_stim
            )
            act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
            spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan
            records.append(
                {
                    "experiment": "timestep_sensitivity",
                    "alpha": alpha,
                    "stimulus_fraction": 0.25,
                    "lesion_fraction": 0.20,
                    "lesion_strategy": "hub",
                    "random_seed": "NA",
                    "dt": dt_val,
                    "network_selection": "highest_degree",
                    "network_seed": 42,
                    "initial_neurons": base_n,
                    "surviving_neurons": les_g_h20.number_of_nodes(),
                    "initial_edges": base_e,
                    "surviving_edges": les_g_h20.number_of_edges(),
                    "edge_retention": round(les_g_h20.number_of_edges() / base_e, 4),
                    "initial_total_synaptic_weight": round(base_w, 1),
                    "surviving_total_synaptic_weight": round(sum(d.get("weight", 1.0) for _, _, d in les_g_h20.edges(data=True)), 1),
                    "synaptic_weight_retention": round(sum(d.get("weight", 1.0) for _, _, d in les_g_h20.edges(data=True)) / base_w, 4),
                    "surviving_stimulated_neurons": len(surv_stim),
                    "is_stable": res.is_stable,
                    "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                    "total_spikes": res.metrics["total_spikes"],
                    "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                    "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                    "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                    "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                    "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                    "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                    "persistence_ratio": rec_data["post_stimulus_persistence"],
                    "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                    "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                    "runtime_seconds": round(dur, 4),
                }
            )

            # 4. Random 20%, seed 42
            les_g_r20, _ = apply_random_lesion(sub_graph, lesion_fraction=0.20, seed=42)
            surv_stim = [nid for nid in stim_ids_25 if nid in les_g_r20]
            res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                graph=les_g_r20, config=cfg, stimulated_neuron_ids=surv_stim
            )
            act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
            spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan
            records.append(
                {
                    "experiment": "timestep_sensitivity",
                    "alpha": alpha,
                    "stimulus_fraction": 0.25,
                    "lesion_fraction": 0.20,
                    "lesion_strategy": "random",
                    "random_seed": 42,
                    "dt": dt_val,
                    "network_selection": "highest_degree",
                    "network_seed": 42,
                    "initial_neurons": base_n,
                    "surviving_neurons": les_g_r20.number_of_nodes(),
                    "initial_edges": base_e,
                    "surviving_edges": les_g_r20.number_of_edges(),
                    "edge_retention": round(les_g_r20.number_of_edges() / base_e, 4),
                    "initial_total_synaptic_weight": round(base_w, 1),
                    "surviving_total_synaptic_weight": round(sum(d.get("weight", 1.0) for _, _, d in les_g_r20.edges(data=True)), 1),
                    "synaptic_weight_retention": round(sum(d.get("weight", 1.0) for _, _, d in les_g_r20.edges(data=True)) / base_w, 4),
                    "surviving_stimulated_neurons": len(surv_stim),
                    "is_stable": res.is_stable,
                    "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                    "total_spikes": res.metrics["total_spikes"],
                    "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                    "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                    "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                    "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                    "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                    "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                    "persistence_ratio": rec_data["post_stimulus_persistence"],
                    "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                    "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                    "runtime_seconds": round(dur, 4),
                }
            )

            logger.info(f"Completed Timestep Sensitivity for alpha={alpha:.4f}, dt={dt_val:.2f} ms (4 simulations)")

    return records


def run_experiment_7e(
    sub_graph_primary: nx.DiGraph,
    sub_graph_random: nx.DiGraph,
    base_config: LIFConfig,
) -> List[Dict[str, Any]]:
    """
    Experiment 7E: Subnetwork Selection Sensitivity
    Compares primary network (highest_degree, seed 42) vs random subnetwork (seed 2026).
    For each network:
    alpha in [0.0200, 0.0300]
    conditions: intact, hub 20%, random 20% (seed 42).
    Total runs: 2 networks x 2 alphas x 3 conditions = 12 simulations.
    """
    logger.info("\n" + "=" * 65)
    logger.info(" EXPERIMENT 7E: Subnetwork Selection Sensitivity")
    logger.info("=" * 65)
    alphas = BASE_ALPHA_VALUES
    records = []

    networks = [
        ("highest_degree", 42, sub_graph_primary),
        ("random", 2026, sub_graph_random),
    ]

    for net_name, net_seed, graph in networks:
        base_n = graph.number_of_nodes()
        base_e = graph.number_of_edges()
        base_w = sum(d.get("weight", 1.0) for _, _, d in graph.edges(data=True))

        stim_ids_25 = select_stimulated_neurons(graph, stimulus_fraction=0.25, seed=42)

        for alpha in alphas:
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

            # 1. Intact baseline
            res_intact, win_intact, rec_intact, _, dur_intact = run_single_sweep_simulation(
                graph=graph,
                config=cfg,
                stimulated_neuron_ids=stim_ids_25,
            )
            base_rate = res_intact.metrics["mean_firing_rate_hz"]
            base_spikes = res_intact.metrics["total_spikes"]

            intact_rec = {
                "experiment": "subnetwork_sensitivity",
                "alpha": alpha,
                "stimulus_fraction": 0.25,
                "lesion_fraction": 0.0,
                "lesion_strategy": "intact",
                "random_seed": "NA",
                "dt": base_config.dt,
                "network_selection": net_name,
                "network_seed": net_seed,
                "initial_neurons": base_n,
                "surviving_neurons": base_n,
                "initial_edges": base_e,
                "surviving_edges": base_e,
                "edge_retention": 1.0,
                "initial_total_synaptic_weight": round(base_w, 1),
                "surviving_total_synaptic_weight": round(base_w, 1),
                "synaptic_weight_retention": 1.0,
                "surviving_stimulated_neurons": len(stim_ids_25),
                "is_stable": res_intact.is_stable,
                "v_max_observed": res_intact.metrics.get("v_max_observed", np.nan),
                "total_spikes": res_intact.metrics["total_spikes"],
                "mean_firing_rate_hz": res_intact.metrics["mean_firing_rate_hz"],
                "active_neuron_fraction": res_intact.metrics["active_neuron_fraction"],
                "stim_window_rate_hz": win_intact["stimulus"]["mean_firing_rate_hz"],
                "post_stim_rate_hz": win_intact["post_stimulus"]["mean_firing_rate_hz"],
                "recruited_unstimulated_count": rec_intact["recruited_unstimulated_count"],
                "recruited_unstimulated_fraction": rec_intact["recruited_unstimulated_fraction"],
                "persistence_ratio": rec_intact["post_stimulus_persistence"],
                "activity_robustness": 1.0,
                "spike_count_robustness": 1.0,
                "runtime_seconds": round(dur_intact, 4),
            }
            records.append(intact_rec)

            # 2. Hub 20%
            les_g_h20, _ = apply_hub_lesion(graph, lesion_fraction=0.20)
            surv_stim = [nid for nid in stim_ids_25 if nid in les_g_h20]
            res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                graph=les_g_h20, config=cfg, stimulated_neuron_ids=surv_stim
            )
            act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
            spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan
            records.append(
                {
                    "experiment": "subnetwork_sensitivity",
                    "alpha": alpha,
                    "stimulus_fraction": 0.25,
                    "lesion_fraction": 0.20,
                    "lesion_strategy": "hub",
                    "random_seed": "NA",
                    "dt": base_config.dt,
                    "network_selection": net_name,
                    "network_seed": net_seed,
                    "initial_neurons": base_n,
                    "surviving_neurons": les_g_h20.number_of_nodes(),
                    "initial_edges": base_e,
                    "surviving_edges": les_g_h20.number_of_edges(),
                    "edge_retention": round(les_g_h20.number_of_edges() / base_e, 4),
                    "initial_total_synaptic_weight": round(base_w, 1),
                    "surviving_total_synaptic_weight": round(sum(d.get("weight", 1.0) for _, _, d in les_g_h20.edges(data=True)), 1),
                    "synaptic_weight_retention": round(sum(d.get("weight", 1.0) for _, _, d in les_g_h20.edges(data=True)) / base_w, 4),
                    "surviving_stimulated_neurons": len(surv_stim),
                    "is_stable": res.is_stable,
                    "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                    "total_spikes": res.metrics["total_spikes"],
                    "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                    "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                    "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                    "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                    "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                    "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                    "persistence_ratio": rec_data["post_stimulus_persistence"],
                    "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                    "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                    "runtime_seconds": round(dur, 4),
                }
            )

            # 3. Random 20%, seed 42
            les_g_r20, _ = apply_random_lesion(graph, lesion_fraction=0.20, seed=42)
            surv_stim = [nid for nid in stim_ids_25 if nid in les_g_r20]
            res, win_data, rec_data, _, dur = run_single_sweep_simulation(
                graph=les_g_r20, config=cfg, stimulated_neuron_ids=surv_stim
            )
            act_rob = (res.metrics["mean_firing_rate_hz"] / base_rate) if base_rate > 0 else np.nan
            spk_rob = (res.metrics["total_spikes"] / base_spikes) if base_spikes > 0 else np.nan
            records.append(
                {
                    "experiment": "subnetwork_sensitivity",
                    "alpha": alpha,
                    "stimulus_fraction": 0.25,
                    "lesion_fraction": 0.20,
                    "lesion_strategy": "random",
                    "random_seed": 42,
                    "dt": base_config.dt,
                    "network_selection": net_name,
                    "network_seed": net_seed,
                    "initial_neurons": base_n,
                    "surviving_neurons": les_g_r20.number_of_nodes(),
                    "initial_edges": base_e,
                    "surviving_edges": les_g_r20.number_of_edges(),
                    "edge_retention": round(les_g_r20.number_of_edges() / base_e, 4),
                    "initial_total_synaptic_weight": round(base_w, 1),
                    "surviving_total_synaptic_weight": round(sum(d.get("weight", 1.0) for _, _, d in les_g_r20.edges(data=True)), 1),
                    "synaptic_weight_retention": round(sum(d.get("weight", 1.0) for _, _, d in les_g_r20.edges(data=True)) / base_w, 4),
                    "surviving_stimulated_neurons": len(surv_stim),
                    "is_stable": res.is_stable,
                    "v_max_observed": res.metrics.get("v_max_observed", np.nan),
                    "total_spikes": res.metrics["total_spikes"],
                    "mean_firing_rate_hz": res.metrics["mean_firing_rate_hz"],
                    "active_neuron_fraction": res.metrics["active_neuron_fraction"],
                    "stim_window_rate_hz": win_data["stimulus"]["mean_firing_rate_hz"],
                    "post_stim_rate_hz": win_data["post_stimulus"]["mean_firing_rate_hz"],
                    "recruited_unstimulated_count": rec_data["recruited_unstimulated_count"],
                    "recruited_unstimulated_fraction": rec_data["recruited_unstimulated_fraction"],
                    "persistence_ratio": rec_data["post_stimulus_persistence"],
                    "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                    "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                    "runtime_seconds": round(dur, 4),
                }
            )

            logger.info(f"Completed Subnetwork Sensitivity for {net_name} (seed {net_seed}), alpha={alpha:.4f} (3 simulations)")

    return records


# ==============================================================================
# FIGURE GENERATION
# ==============================================================================

def generate_phase7_figures(
    df_7a: pd.DataFrame,
    df_7b: pd.DataFrame,
    df_7c: pd.DataFrame,
    df_7d: pd.DataFrame,
    df_7e: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Generates 5 publication-oriented figures for Phase 7.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # 1. Phase 7A: Local Alpha Sensitivity
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    color1 = "#1f77b4"
    color2 = "#d62728"
    color3 = "#2ca02c"

    ax1.plot(df_7a["alpha"], df_7a["mean_firing_rate_hz"], marker="o", color=color1, linewidth=2.2, label="Mean Firing Rate (Hz)")
    ax1.plot(df_7a["alpha"], df_7a["post_stim_rate_hz"], marker="s", color=color2, linewidth=2.2, linestyle="--", label="Post-Stimulus Rate (Hz)")
    ax1.set_xlabel(r"Coupling Scale $\alpha$ (mV / synapse contact)", fontweight="bold")
    ax1.set_ylabel("Firing Rate (Hz)", fontweight="bold", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(df_7a["alpha"], df_7a["recruited_unstimulated_fraction"] * 100, marker="^", color=color3, linewidth=2.0, linestyle="-.", label="Recruited Unstim. (%)")
    ax2.set_ylabel("Recruited Unstimulated Neurons (%)", fontweight="bold", color=color3)
    ax2.tick_params(axis="y", labelcolor=color3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    plt.title("Experiment 7A: Local Dynamic Transition Across Nearby Coupling Scales", fontweight="bold", pad=12)
    p7a_fig = figures_dir / "phase7_alpha_sensitivity.png"
    plt.savefig(p7a_fig, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {p7a_fig}")

    # 2. Phase 7B: Stimulus-Fraction Sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)

    for idx, (alpha_val, ax) in enumerate([(0.0200, ax1), (0.0300, ax2)]):
        df_sub = df_7b[df_7b["alpha"] == alpha_val]
        stim_fractions = [0.05, 0.10, 0.25]
        x = np.arange(len(stim_fractions))
        width = 0.22

        # Hub 10% and 20%
        hub10_vals = [df_sub[(df_sub["stimulus_fraction"] == sf) & (df_sub["lesion_strategy"] == "hub") & (df_sub["lesion_fraction"] == 0.10)]["activity_robustness"].values[0] for sf in stim_fractions]
        hub20_vals = [df_sub[(df_sub["stimulus_fraction"] == sf) & (df_sub["lesion_strategy"] == "hub") & (df_sub["lesion_fraction"] == 0.20)]["activity_robustness"].values[0] for sf in stim_fractions]

        # Random 10% and 20% (mean and sd)
        rnd10_means, rnd10_sds = [], []
        rnd20_means, rnd20_sds = [], []
        for sf in stim_fractions:
            r10_data = df_sub[(df_sub["stimulus_fraction"] == sf) & (df_sub["lesion_strategy"] == "random") & (df_sub["lesion_fraction"] == 0.10)]["activity_robustness"]
            rnd10_means.append(r10_data.mean())
            rnd10_sds.append(r10_data.std())

            r20_data = df_sub[(df_sub["stimulus_fraction"] == sf) & (df_sub["lesion_strategy"] == "random") & (df_sub["lesion_fraction"] == 0.20)]["activity_robustness"]
            rnd20_means.append(r20_data.mean())
            rnd20_sds.append(r20_data.std())

        ax.bar(x - 1.5 * width, hub10_vals, width, label="Hub 10%", color="#d95f02", alpha=0.9)
        ax.bar(x - 0.5 * width, hub20_vals, width, label="Hub 20%", color="#a63603", alpha=0.9)
        ax.bar(x + 0.5 * width, rnd10_means, width, yerr=rnd10_sds, capsize=4, label="Random 10% (Mean ± SD)", color="#7570b3", alpha=0.9)
        ax.bar(x + 1.5 * width, rnd20_means, width, yerr=rnd20_sds, capsize=4, label="Random 20% (Mean ± SD)", color="#1b9e77", alpha=0.9)

        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
        ax.set_title(rf"Coupling $\alpha = {alpha_val:.4f}$ mV/synapse", fontweight="bold")
        ax.set_xlabel("Stimulus Fraction", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["5%", "10%", "25%"])
        ax.set_ylim(0, 1.25)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel(r"Activity Robustness ($R_{\mathrm{act}}$)", fontweight="bold")
            ax.legend(frameon=True, loc="upper right")

    plt.suptitle("Experiment 7B: Activity Robustness Under Varying Stimulus Drive", fontweight="bold", y=1.02)
    p7b_fig = figures_dir / "phase7_stimulus_sensitivity.png"
    plt.savefig(p7b_fig, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {p7b_fig}")

    # 3. Phase 7C: Lesion-Fraction Sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)

    for idx, (alpha_val, ax) in enumerate([(0.0200, ax1), (0.0300, ax2)]):
        df_sub = df_7c[df_7c["alpha"] == alpha_val]
        lfs = [0.0, 0.05, 0.10, 0.20]

        hub_pts = [1.0]
        for lf in [0.05, 0.10, 0.20]:
            v = df_sub[(df_sub["lesion_fraction"] == lf) & (df_sub["lesion_strategy"] == "hub")]["activity_robustness"].values[0]
            hub_pts.append(v)

        rnd_means = [1.0]
        rnd_sds = [0.0]
        for lf in [0.05, 0.10, 0.20]:
            r_data = df_sub[(df_sub["lesion_fraction"] == lf) & (df_sub["lesion_strategy"] == "random")]["activity_robustness"]
            rnd_means.append(r_data.mean())
            rnd_sds.append(r_data.std())

        lf_pct = [l * 100 for l in lfs]
        ax.plot(lf_pct, hub_pts, marker="o", linewidth=2.2, color="#d95f02", label="Targeted Hub Lesion")
        ax.errorbar(lf_pct, rnd_means, yerr=rnd_sds, marker="s", linewidth=2.2, color="#1b9e77", capsize=4, label="Random Lesion (Mean ± SD)")

        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
        ax.set_title(rf"Coupling $\alpha = {alpha_val:.4f}$ mV/synapse", fontweight="bold")
        ax.set_xlabel("Lesion Fraction (%)", fontweight="bold")
        ax.set_xticks(lf_pct)
        ax.set_ylim(0, 1.15)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel(r"Activity Robustness ($R_{\mathrm{act}}$)", fontweight="bold")
            ax.legend(frameon=True, loc="lower left")

    plt.suptitle("Experiment 7C: Activity Robustness Scaling Across Lesion Fractions", fontweight="bold", y=1.02)
    p7c_fig = figures_dir / "phase7_lesion_fraction_sensitivity.png"
    plt.savefig(p7c_fig, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {p7c_fig}")

    # 4. Phase 7D: Timestep Sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)

    for idx, (alpha_val, ax) in enumerate([(0.0200, ax1), (0.0300, ax2)]):
        df_sub = df_7d[df_7d["alpha"] == alpha_val]
        dts = [0.25, 0.50, 1.00]
        x = np.arange(len(dts))
        width = 0.2

        intact_rates = [df_sub[(df_sub["dt"] == d) & (df_sub["lesion_strategy"] == "intact")]["mean_firing_rate_hz"].values[0] for d in dts]
        hub10_rates = [df_sub[(df_sub["dt"] == d) & (df_sub["lesion_strategy"] == "hub") & (df_sub["lesion_fraction"] == 0.10)]["mean_firing_rate_hz"].values[0] for d in dts]
        hub20_rates = [df_sub[(df_sub["dt"] == d) & (df_sub["lesion_strategy"] == "hub") & (df_sub["lesion_fraction"] == 0.20)]["mean_firing_rate_hz"].values[0] for d in dts]
        rnd20_rates = [df_sub[(df_sub["dt"] == d) & (df_sub["lesion_strategy"] == "random") & (df_sub["lesion_fraction"] == 0.20)]["mean_firing_rate_hz"].values[0] for d in dts]

        ax.bar(x - 1.5 * width, intact_rates, width, label="Intact Baseline", color="#2b83ba", alpha=0.9)
        ax.bar(x - 0.5 * width, hub10_rates, width, label="Hub 10%", color="#fdae61", alpha=0.9)
        ax.bar(x + 0.5 * width, hub20_rates, width, label="Hub 20%", color="#d7191c", alpha=0.9)
        ax.bar(x + 1.5 * width, rnd20_rates, width, label="Random 20% (Seed 42)", color="#abdda4", alpha=0.9)

        ax.set_title(rf"Coupling $\alpha = {alpha_val:.4f}$ mV/synapse", fontweight="bold")
        ax.set_xlabel(r"Integration Timestep $dt$ (ms)", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["0.25 ms", "0.50 ms", "1.00 ms"])
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Mean Firing Rate (Hz)", fontweight="bold")
            ax.legend(frameon=True, loc="upper right")

    plt.suptitle("Experiment 7D: Simulation Invariance Across Integration Timesteps", fontweight="bold", y=1.02)
    p7d_fig = figures_dir / "phase7_timestep_sensitivity.png"
    plt.savefig(p7d_fig, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {p7d_fig}")

    # 5. Phase 7E: Subnetwork Selection Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)

    for idx, (alpha_val, ax) in enumerate([(0.0200, ax1), (0.0300, ax2)]):
        df_sub = df_7e[df_7e["alpha"] == alpha_val]
        nets = ["highest_degree", "random"]
        x = np.arange(len(nets))
        width = 0.25

        intact_vals = [df_sub[(df_sub["network_selection"] == n) & (df_sub["lesion_strategy"] == "intact")]["activity_robustness"].values[0] for n in nets]
        hub20_vals = [df_sub[(df_sub["network_selection"] == n) & (df_sub["lesion_strategy"] == "hub") & (df_sub["lesion_fraction"] == 0.20)]["activity_robustness"].values[0] for n in nets]
        rnd20_vals = [df_sub[(df_sub["network_selection"] == n) & (df_sub["lesion_strategy"] == "random") & (df_sub["lesion_fraction"] == 0.20)]["activity_robustness"].values[0] for n in nets]

        ax.bar(x - width, intact_vals, width, label="Intact Baseline", color="#2b83ba", alpha=0.9)
        ax.bar(x, hub20_vals, width, label="Hub 20% Lesion", color="#d7191c", alpha=0.9)
        ax.bar(x + width, rnd20_vals, width, label="Random 20% Lesion (Seed 42)", color="#1b9e77", alpha=0.9)

        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
        ax.set_title(rf"Coupling $\alpha = {alpha_val:.4f}$ mV/synapse", fontweight="bold")
        ax.set_xlabel("Subnetwork Selection Strategy", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Hub-Enriched Core\n(highest_degree)", "Random Sample\n(seed 2026)"])
        ax.set_ylim(0, 1.2)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel(r"Activity Robustness ($R_{\mathrm{act}}$)", fontweight="bold")
            ax.legend(frameon=True, loc="upper right")

    plt.suptitle("Experiment 7E: Robustness Comparison Across Distinct 1,000-Neuron Subnetworks", fontweight="bold", y=1.02)
    p7e_fig = figures_dir / "phase7_subnetwork_comparison.png"
    plt.savefig(p7e_fig, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {p7e_fig}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 7 Robustness & Sensitivity Validation")
    parser.add_argument("--smoke-test", action="store_true", help="Quick sanity test with fewer seeds")
    args = parser.parse_args()

    t_start = time.perf_counter()
    tracemalloc.start()

    logger.info("=" * 65)
    logger.info(" PHASE 7 — ROBUSTNESS & SENSITIVITY VALIDATION")
    logger.info("=" * 65)

    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Primary Subnetwork (highest_degree, seed 42)
    logger.info("Extracting Primary Subnetwork (highest_degree, 1000 neurons, seed 42)...")
    sub_graph_primary, diag_primary = extract_real_subnetwork(
        max_neurons=1000, strategy="highest_degree", seed=42, verbose=False
    )
    logger.info(
        f"Primary Subnetwork: {sub_graph_primary.number_of_nodes()} neurons, "
        f"{sub_graph_primary.number_of_edges()} directed edges, "
        f"{diag_primary['total_synaptic_weight']:.0f} synapses."
    )

    # 2. Load Secondary Subnetwork (random, seed 2026)
    logger.info("Extracting Secondary Subnetwork (random, 1000 neurons, seed 2026)...")
    sub_graph_random, diag_random = extract_real_subnetwork(
        max_neurons=1000, strategy="random", seed=2026, verbose=False
    )
    sum_rand = get_graph_summary(sub_graph_random)
    logger.info(
        f"Random Subnetwork: {sub_graph_random.number_of_nodes()} neurons, "
        f"{sub_graph_random.number_of_edges()} directed edges, "
        f"{sum_rand['total_synapses']:.0f} synapses, "
        f"Avg degree: {sum_rand['average_degree']:.2f}, Max degree: {sum_rand['max_degree']}, "
        f"Density: {sum_rand['density']:.6f}, WCC: {sum_rand['num_weakly_connected']}, "
        f"Largest WCC: {sum_rand['largest_wcc_size']} ({sum_rand['largest_wcc_fraction']*100:.1f}%)."
    )

    # Pre-select deterministic stimulated neuron sets on primary network
    stim_sets_primary = {
        0.05: select_stimulated_neurons(sub_graph_primary, stimulus_fraction=0.05, seed=42),
        0.10: select_stimulated_neurons(sub_graph_primary, stimulus_fraction=0.10, seed=42),
        0.25: select_stimulated_neurons(sub_graph_primary, stimulus_fraction=0.25, seed=42),
    }

    base_config = LIFConfig(
        tau_m=20.0,
        v_rest=-65.0,
        v_reset=-70.0,
        v_threshold=-50.0,
        t_ref=2.0,
        dt=0.5,
        duration=1000.0,
        external_current=18.0,
        noise_sigma=1.0,
        pulse_start=200.0,
        pulse_end=600.0,
        random_seed=42,
    )

    # --------------------------------------------------------------------------
    # Execute Experiment 7A (5 simulations)
    # --------------------------------------------------------------------------
    recs_7a = run_experiment_7a(sub_graph_primary, stim_sets_primary[0.25], base_config)
    df_7a = pd.DataFrame(recs_7a)
    p7a_csv = tables_dir / "phase7_alpha_sensitivity.csv"
    df_7a.to_csv(p7a_csv, index=False)
    logger.info(f"Saved: {p7a_csv} ({len(df_7a)} rows)")

    # --------------------------------------------------------------------------
    # Execute Experiment 7B (78 simulations)
    # --------------------------------------------------------------------------
    recs_7b = run_experiment_7b(sub_graph_primary, stim_sets_primary, base_config)
    df_7b = pd.DataFrame(recs_7b)
    p7b_csv = tables_dir / "phase7_stimulus_sensitivity.csv"
    df_7b.to_csv(p7b_csv, index=False)
    logger.info(f"Saved: {p7b_csv} ({len(df_7b)} rows)")

    # --------------------------------------------------------------------------
    # Execute Experiment 7C (38 simulations)
    # --------------------------------------------------------------------------
    recs_7c = run_experiment_7c(sub_graph_primary, stim_sets_primary[0.25], base_config)
    df_7c = pd.DataFrame(recs_7c)
    p7c_csv = tables_dir / "phase7_lesion_sensitivity.csv"
    df_7c.to_csv(p7c_csv, index=False)
    logger.info(f"Saved: {p7c_csv} ({len(df_7c)} rows)")

    # --------------------------------------------------------------------------
    # Execute Experiment 7D (24 simulations)
    # --------------------------------------------------------------------------
    recs_7d = run_experiment_7d(sub_graph_primary, stim_sets_primary[0.25], base_config)
    df_7d = pd.DataFrame(recs_7d)
    p7d_csv = tables_dir / "phase7_timestep_sensitivity.csv"
    df_7d.to_csv(p7d_csv, index=False)
    logger.info(f"Saved: {p7d_csv} ({len(df_7d)} rows)")

    # --------------------------------------------------------------------------
    # Execute Experiment 7E (12 simulations)
    # --------------------------------------------------------------------------
    recs_7e = run_experiment_7e(sub_graph_primary, sub_graph_random, base_config)
    df_7e = pd.DataFrame(recs_7e)
    p7e_csv = tables_dir / "phase7_subnetwork_sensitivity.csv"
    df_7e.to_csv(p7e_csv, index=False)
    logger.info(f"Saved: {p7e_csv} ({len(df_7e)} rows)")

    # --------------------------------------------------------------------------
    # Master Summary Table
    # --------------------------------------------------------------------------
    all_recs = recs_7a + recs_7b + recs_7c + recs_7d + recs_7e
    df_summary = pd.DataFrame(all_recs)
    psum_csv = tables_dir / "phase7_summary.csv"
    df_summary.to_csv(psum_csv, index=False)
    logger.info(f"Saved Master Phase 7 Summary Table: {psum_csv} ({len(df_summary)} rows)")

    # --------------------------------------------------------------------------
    # Generate Publication Figures
    # --------------------------------------------------------------------------
    logger.info("\nGenerating publication figures for Phase 7...")
    generate_phase7_figures(df_7a, df_7b, df_7c, df_7d, df_7e, figures_dir)

    t_end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info("=" * 65)
    logger.info(" PHASE 7 EXECUTION COMPLETE")
    logger.info(f" Total Simulations: {len(all_recs)}")
    logger.info(f" - 7A Alpha Sensitivity:       {len(recs_7a)}")
    logger.info(f" - 7B Stimulus Sensitivity:    {len(recs_7b)}")
    logger.info(f" - 7C Lesion Sensitivity:      {len(recs_7c)}")
    logger.info(f" - 7D Timestep Sensitivity:    {len(recs_7d)}")
    logger.info(f" - 7E Subnetwork Sensitivity:  {len(recs_7e)}")
    logger.info(f" Total Runtime:     {t_end - t_start:.2f} s")
    logger.info(f" Peak Memory:       {peak_mem / (1024*1024):.2f} MB")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
