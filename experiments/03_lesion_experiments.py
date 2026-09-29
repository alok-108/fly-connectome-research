"""
Experiment 03: Virtual Lesion Analysis of Drosophila Connectome Subnetwork (Phase 4).

Performs systematic virtual lesion experiments on the real 1,000-neuron
Janelia FlyEM Hemibrain v1.2.1 connectome subnetwork:
- Condition A: Baseline (intact network)
- Condition B: Random lesions (1%, 5%, 10%, 20% ablated across 5 seeds: 42, 123, 456, 789, 1000)
- Condition C: Targeted hub lesions (1%, 5%, 10%, 20% ablated, ranked by total pre-lesion degree)

Measures structural robustness (remaining edges, synaptic weights, WCC) and functional
activity dynamics (firing rates, spike counts, active fraction, population rate).
Generates publication-quality figures and comprehensive CSV summaries.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.data_loader import extract_real_subnetwork
from src.lesion import (
    apply_hub_lesion,
    apply_random_lesion,
    compute_lesion_graph_metrics,
)
from src.simulation import LIFConfig, LIFNetwork, SimulationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_single_condition(
    network_graph: nx.DiGraph,
    config: LIFConfig,
    strategy: str,
    fraction: float,
    seed: Optional[int],
    baseline_metrics: Optional[Dict[str, Any]] = None,
    ablated_nodes: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], SimulationResult]:
    """
    Simulates a single experimental condition and calculates all metrics.

    Parameters:
        network_graph: Intact or lesioned DiGraph.
        config: LIFConfig for simulation.
        strategy: 'baseline', 'random', or 'hub'.
        fraction: Fraction of neurons removed (0.0 for baseline).
        seed: Random seed for lesion selection (or None for hub/baseline).
        baseline_metrics: Reference metrics from intact baseline run for relative changes.
        ablated_nodes: List of neuron IDs removed in this condition.

    Returns:
        (raw_record, graph_metrics_record, simulation_result)
    """
    total_nodes = network_graph.number_of_nodes()
    total_edges = network_graph.number_of_edges()
    total_weight = sum(d.get("weight", 1.0) for _, _, d in network_graph.edges(data=True))

    wcc_components = list(nx.weakly_connected_components(network_graph))
    largest_wcc = max(len(c) for c in wcc_components) if total_nodes > 0 else 0

    # Execute LIF Simulation
    network = LIFNetwork(network_graph)
    t_start = time.perf_counter()
    sim_res = network.simulate(config, record_samples=0)
    runtime_sec = time.perf_counter() - t_start

    sim_m = sim_res.metrics
    mean_rate = float(sim_m["mean_firing_rate_hz"])
    median_rate = float(sim_m["median_firing_rate_hz"])
    max_rate = float(sim_m["max_firing_rate_hz"])
    active_fraction = float(sim_m["active_neuron_fraction"])
    total_spikes = int(sim_m["total_spikes"])

    pop_rates = sim_res.population_rate
    pop_peak = float(np.max(pop_rates)) if len(pop_rates) > 0 else 0.0
    pop_mean = float(np.mean(pop_rates)) if len(pop_rates) > 0 else 0.0

    # Compute baseline-relative metrics
    if baseline_metrics is None:
        # This is the baseline run itself
        rel_change_rate = 0.0
        rel_change_spikes = 0.0
        activity_robustness = 1.0
        network_robustness = 1.0
        rem_edge_fraction = 1.0
        rem_weight_fraction = 1.0
        largest_wcc_rel_fraction = 1.0
        base_neurons = total_nodes
        base_edges = total_edges
        base_weight = total_weight
        base_wcc = largest_wcc
        lesion_count = 0
    else:
        base_rate = baseline_metrics["mean_firing_rate_hz"]
        base_spikes = baseline_metrics["total_spikes"]
        base_edges = baseline_metrics["baseline_edges"]
        base_weight = baseline_metrics["baseline_weight"]
        base_wcc = baseline_metrics["baseline_wcc"]
        base_neurons = baseline_metrics["baseline_neurons"]

        lesion_count = base_neurons - total_nodes

        rel_change_rate = ((mean_rate - base_rate) / base_rate) if base_rate > 0 else 0.0
        rel_change_spikes = ((total_spikes - base_spikes) / base_spikes) if base_spikes > 0 else 0.0
        activity_robustness = (mean_rate / base_rate) if base_rate > 0 else 0.0
        network_robustness = (total_edges / base_edges) if base_edges > 0 else 0.0
        rem_edge_fraction = (total_edges / base_edges) if base_edges > 0 else 0.0
        rem_weight_fraction = (total_weight / base_weight) if base_weight > 0 else 0.0
        largest_wcc_rel_fraction = (largest_wcc / base_wcc) if base_wcc > 0 else 0.0

    ablated_str = ";".join(str(nid) for nid in (ablated_nodes or []))

    raw_record = {
        "lesion_strategy": strategy,
        "lesion_fraction": fraction,
        "lesion_count": lesion_count,
        "seed": seed if seed is not None else "NA",
        "remaining_neurons": total_nodes,
        "remaining_edges": total_edges,
        "remaining_total_synaptic_weight": round(total_weight, 1),
        "largest_weak_component_size": largest_wcc,
        "largest_wcc_relative_fraction": round(largest_wcc_rel_fraction, 4),
        "total_spikes": total_spikes,
        "mean_firing_rate_hz": round(mean_rate, 4),
        "median_firing_rate_hz": round(median_rate, 4),
        "max_firing_rate_hz": round(max_rate, 4),
        "active_neuron_fraction": round(active_fraction, 4),
        "population_peak_rate_hz": round(pop_peak, 4),
        "population_mean_rate_hz": round(pop_mean, 4),
        "relative_change_from_baseline": round(rel_change_rate, 4),
        "relative_change_total_spikes": round(rel_change_spikes, 4),
        "network_robustness": round(network_robustness, 4),
        "activity_robustness": round(activity_robustness, 4),
        "remaining_edge_fraction": round(rem_edge_fraction, 4),
        "remaining_weight_fraction": round(rem_weight_fraction, 4),
        "simulation_stable": sim_res.is_stable,
        "runtime_seconds": round(runtime_sec, 4),
        "ablated_neuron_ids": ablated_str,
    }

    graph_record = {
        "lesion_strategy": strategy,
        "lesion_fraction": fraction,
        "lesion_count": lesion_count,
        "seed": seed if seed is not None else "NA",
        "baseline_neurons": base_neurons,
        "remaining_neurons": total_nodes,
        "baseline_edges": base_edges,
        "remaining_edges": total_edges,
        "remaining_total_synaptic_weight": round(total_weight, 1),
        "largest_weak_component_size": largest_wcc,
        "largest_wcc_relative_fraction": round(largest_wcc_rel_fraction, 4),
        "network_robustness": round(network_robustness, 4),
        "remaining_edge_fraction": round(rem_edge_fraction, 4),
        "remaining_weight_fraction": round(rem_weight_fraction, 4),
        "num_weak_components": len(wcc_components),
    }

    return raw_record, graph_record, sim_res


def generate_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes aggregated descriptive statistics (mean, std, min, max) for random lesions,
    and formats baseline and hub conditions.
    """
    summary_records = []

    # 1. Baseline condition
    base_rows = raw_df[raw_df["lesion_strategy"] == "baseline"]
    for _, row in base_rows.iterrows():
        summary_records.append({
            "lesion_strategy": "baseline",
            "lesion_fraction": 0.0,
            "lesion_count": 0,
            "num_runs": 1,
            "mean_firing_rate_mean": row["mean_firing_rate_hz"],
            "mean_firing_rate_std": 0.0,
            "mean_firing_rate_min": row["mean_firing_rate_hz"],
            "mean_firing_rate_max": row["mean_firing_rate_hz"],
            "total_spikes_mean": row["total_spikes"],
            "total_spikes_std": 0.0,
            "total_spikes_min": row["total_spikes"],
            "total_spikes_max": row["total_spikes"],
            "active_neuron_fraction_mean": row["active_neuron_fraction"],
            "active_neuron_fraction_std": 0.0,
            "network_robustness_mean": 1.0,
            "network_robustness_std": 0.0,
            "activity_robustness_mean": 1.0,
            "activity_robustness_std": 0.0,
            "relative_change_rate_mean": 0.0,
            "relative_change_rate_std": 0.0,
            "remaining_edges_mean": row["remaining_edges"],
            "remaining_edges_std": 0.0,
            "remaining_weight_mean": row["remaining_total_synaptic_weight"],
            "remaining_weight_std": 0.0,
            "largest_wcc_relative_fraction_mean": 1.0,
            "largest_wcc_relative_fraction_std": 0.0,
        })

    # 2. Hub conditions (deterministic, 1 run per fraction)
    hub_rows = raw_df[raw_df["lesion_strategy"] == "hub"]
    for frac in sorted(hub_rows["lesion_fraction"].unique()):
        sub = hub_rows[hub_rows["lesion_fraction"] == frac].iloc[0]
        summary_records.append({
            "lesion_strategy": "hub",
            "lesion_fraction": frac,
            "lesion_count": int(sub["lesion_count"]),
            "num_runs": 1,
            "mean_firing_rate_mean": sub["mean_firing_rate_hz"],
            "mean_firing_rate_std": 0.0,
            "mean_firing_rate_min": sub["mean_firing_rate_hz"],
            "mean_firing_rate_max": sub["mean_firing_rate_hz"],
            "total_spikes_mean": sub["total_spikes"],
            "total_spikes_std": 0.0,
            "total_spikes_min": sub["total_spikes"],
            "total_spikes_max": sub["total_spikes"],
            "active_neuron_fraction_mean": sub["active_neuron_fraction"],
            "active_neuron_fraction_std": 0.0,
            "network_robustness_mean": sub["network_robustness"],
            "network_robustness_std": 0.0,
            "activity_robustness_mean": sub["activity_robustness"],
            "activity_robustness_std": 0.0,
            "relative_change_rate_mean": sub["relative_change_from_baseline"],
            "relative_change_rate_std": 0.0,
            "remaining_edges_mean": sub["remaining_edges"],
            "remaining_edges_std": 0.0,
            "remaining_weight_mean": sub["remaining_total_synaptic_weight"],
            "remaining_weight_std": 0.0,
            "largest_wcc_relative_fraction_mean": sub["largest_wcc_relative_fraction"],
            "largest_wcc_relative_fraction_std": 0.0,
        })

    # 3. Random conditions (aggregated across 5 seeds per fraction)
    rnd_rows = raw_df[raw_df["lesion_strategy"] == "random"]
    for frac in sorted(rnd_rows["lesion_fraction"].unique()):
        sub = rnd_rows[rnd_rows["lesion_fraction"] == frac]
        n_runs = len(sub)
        summary_records.append({
            "lesion_strategy": "random",
            "lesion_fraction": frac,
            "lesion_count": int(sub["lesion_count"].iloc[0]),
            "num_runs": n_runs,
            "mean_firing_rate_mean": round(float(sub["mean_firing_rate_hz"].mean()), 4),
            "mean_firing_rate_std": round(float(sub["mean_firing_rate_hz"].std()), 4),
            "mean_firing_rate_min": round(float(sub["mean_firing_rate_hz"].min()), 4),
            "mean_firing_rate_max": round(float(sub["mean_firing_rate_hz"].max()), 4),
            "total_spikes_mean": round(float(sub["total_spikes"].mean()), 1),
            "total_spikes_std": round(float(sub["total_spikes"].std()), 1),
            "total_spikes_min": int(sub["total_spikes"].min()),
            "total_spikes_max": int(sub["total_spikes"].max()),
            "active_neuron_fraction_mean": round(float(sub["active_neuron_fraction"].mean()), 4),
            "active_neuron_fraction_std": round(float(sub["active_neuron_fraction"].std()), 4),
            "network_robustness_mean": round(float(sub["network_robustness"].mean()), 4),
            "network_robustness_std": round(float(sub["network_robustness"].std()), 4),
            "activity_robustness_mean": round(float(sub["activity_robustness"].mean()), 4),
            "activity_robustness_std": round(float(sub["activity_robustness"].std()), 4),
            "relative_change_rate_mean": round(float(sub["relative_change_from_baseline"].mean()), 4),
            "relative_change_rate_std": round(float(sub["relative_change_from_baseline"].std()), 4),
            "remaining_edges_mean": round(float(sub["remaining_edges"].mean()), 1),
            "remaining_edges_std": round(float(sub["remaining_edges"].std()), 1),
            "remaining_weight_mean": round(float(sub["remaining_total_synaptic_weight"].mean()), 1),
            "remaining_weight_std": round(float(sub["remaining_total_synaptic_weight"].std()), 1),
            "largest_wcc_relative_fraction_mean": round(float(sub["largest_wcc_relative_fraction"].mean()), 4),
            "largest_wcc_relative_fraction_std": round(float(sub["largest_wcc_relative_fraction"].std()), 4),
        })

    return pd.DataFrame(summary_records)


def generate_lesion_figures(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Renders 5 publication-quality scientific figures comparing random vs hub lesions.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("tableau-colorblind10")

    # Data extracts
    base_row = summary_df[summary_df["lesion_strategy"] == "baseline"].iloc[0]
    base_rate = base_row["mean_firing_rate_mean"]
    base_spikes = base_row["total_spikes_mean"]
    base_active = base_row["active_neuron_fraction_mean"] * 100.0

    hub_df = summary_df[summary_df["lesion_strategy"] == "hub"].sort_values("lesion_fraction")
    rnd_df = summary_df[summary_df["lesion_strategy"] == "random"].sort_values("lesion_fraction")

    # Add 0% baseline point to curves for continuous comparison
    fractions_hub = np.array([0.0] + list(hub_df["lesion_fraction"].values)) * 100.0
    fractions_rnd = np.array([0.0] + list(rnd_df["lesion_fraction"].values)) * 100.0

    rates_hub = np.array([base_rate] + list(hub_df["mean_firing_rate_mean"].values))
    rates_rnd = np.array([base_rate] + list(rnd_df["mean_firing_rate_mean"].values))
    rates_rnd_std = np.array([0.0] + list(rnd_df["mean_firing_rate_std"].values))

    spikes_hub = np.array([base_spikes] + list(hub_df["total_spikes_mean"].values))
    spikes_rnd = np.array([base_spikes] + list(rnd_df["total_spikes_mean"].values))
    spikes_rnd_std = np.array([0.0] + list(rnd_df["total_spikes_std"].values))

    active_hub = np.array([base_active] + list(hub_df["active_neuron_fraction_mean"].values * 100.0))
    active_rnd = np.array([base_active] + list(rnd_df["active_neuron_fraction_mean"].values * 100.0))
    active_rnd_std = np.array([0.0] + list(rnd_df["active_neuron_fraction_std"].values * 100.0))

    edges_hub = np.array([1.0] + list(hub_df["network_robustness_mean"].values)) * 100.0
    edges_rnd = np.array([1.0] + list(rnd_df["network_robustness_mean"].values)) * 100.0
    edges_rnd_std = np.array([0.0] + list(rnd_df["network_robustness_std"].values)) * 100.0

    weights_hub = np.array([1.0] + list(hub_df["remaining_weight_mean"].values / base_row["remaining_weight_mean"])) * 100.0
    weights_rnd = np.array([1.0] + list(rnd_df["remaining_weight_mean"].values / base_row["remaining_weight_mean"])) * 100.0
    weights_rnd_std = np.array([0.0] + list(rnd_df["remaining_weight_std"].values / base_row["remaining_weight_mean"])) * 100.0

    color_rnd = "#1f77b4"  # Blue
    color_hub = "#d62728"  # Red
    color_base = "#2ca02c"  # Green

    # =========================================================================
    # Figure 1: Mean Firing Rate Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Random lesion with std error band
    ax.errorbar(
        fractions_rnd,
        rates_rnd,
        yerr=rates_rnd_std,
        fmt="o-",
        linewidth=2.0,
        markersize=6,
        color=color_rnd,
        ecolor=color_rnd,
        capsize=4,
        capthick=1.2,
        label="Random Lesion (Mean ± SD, 5 seeds)",
    )
    ax.fill_between(
        fractions_rnd,
        rates_rnd - rates_rnd_std,
        rates_rnd + rates_rnd_std,
        color=color_rnd,
        alpha=0.15,
    )

    # Hub lesion deterministic curve
    ax.plot(
        fractions_hub,
        rates_hub,
        "s--",
        linewidth=2.0,
        markersize=6,
        color=color_hub,
        label="Targeted Hub Lesion (Highest-Degree)",
    )

    # Baseline reference line
    ax.axhline(base_rate, color=color_base, linestyle=":", linewidth=1.5, label=f"Baseline ({base_rate:.2f} Hz)")

    ax.set_title("Impact of Virtual Lesions on Population Mean Firing Rate", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Lesion Fraction (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Firing Rate (Hz)", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1, 5, 10, 20])
    ax.set_xlim(-0.5, 21.0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=10, loc="lower left")

    plt.tight_layout()
    fig1_path = figures_dir / "lesion_firing_rate_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 1 to: {fig1_path}")

    # =========================================================================
    # Figure 2: Total Spike Count Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    ax.errorbar(
        fractions_rnd,
        spikes_rnd,
        yerr=spikes_rnd_std,
        fmt="o-",
        linewidth=2.0,
        markersize=6,
        color=color_rnd,
        ecolor=color_rnd,
        capsize=4,
        capthick=1.2,
        label="Random Lesion (Mean ± SD, 5 seeds)",
    )
    ax.fill_between(
        fractions_rnd,
        spikes_rnd - spikes_rnd_std,
        spikes_rnd + spikes_rnd_std,
        color=color_rnd,
        alpha=0.15,
    )

    ax.plot(
        fractions_hub,
        spikes_hub,
        "s--",
        linewidth=2.0,
        markersize=6,
        color=color_hub,
        label="Targeted Hub Lesion (Highest-Degree)",
    )

    ax.axhline(base_spikes, color=color_base, linestyle=":", linewidth=1.5, label=f"Baseline ({int(base_spikes):,} spikes)")

    ax.set_title("Total Population Spike Count Across Lesion Conditions", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Lesion Fraction (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Spikes (1000 ms simulation)", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1, 5, 10, 20])
    ax.set_xlim(-0.5, 21.0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=10, loc="lower left")

    plt.tight_layout()
    fig2_path = figures_dir / "lesion_spike_count_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 2 to: {fig2_path}")

    # =========================================================================
    # Figure 3: Active Neuron Fraction
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    ax.errorbar(
        fractions_rnd,
        active_rnd,
        yerr=active_rnd_std,
        fmt="o-",
        linewidth=2.0,
        markersize=6,
        color=color_rnd,
        ecolor=color_rnd,
        capsize=4,
        capthick=1.2,
        label="Random Lesion (Mean ± SD, 5 seeds)",
    )
    ax.fill_between(
        fractions_rnd,
        active_rnd - active_rnd_std,
        active_rnd + active_rnd_std,
        color=color_rnd,
        alpha=0.15,
    )

    ax.plot(
        fractions_hub,
        active_hub,
        "s--",
        linewidth=2.0,
        markersize=6,
        color=color_hub,
        label="Targeted Hub Lesion (Highest-Degree)",
    )

    ax.axhline(base_active, color=color_base, linestyle=":", linewidth=1.5, label=f"Baseline ({base_active:.1f}%)")

    ax.set_title("Active Neuron Fraction (% Firing ≥1 Spike)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Lesion Fraction (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Active Neurons (%)", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1, 5, 10, 20])
    ax.set_xlim(-0.5, 21.0)
    ax.set_ylim(bottom=0, top=max(base_active * 1.25, 40.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=10, loc="lower left")

    plt.tight_layout()
    fig3_path = figures_dir / "lesion_active_fraction.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 3 to: {fig3_path}")

    # =========================================================================
    # Figure 4: Network Connectivity (Edges & Weights)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel A: Remaining Edges
    ax1.errorbar(
        fractions_rnd,
        edges_rnd,
        yerr=edges_rnd_std,
        fmt="o-",
        linewidth=2.0,
        markersize=6,
        color=color_rnd,
        capsize=4,
        label="Random (Mean ± SD)",
    )
    ax1.fill_between(fractions_rnd, edges_rnd - edges_rnd_std, edges_rnd + edges_rnd_std, color=color_rnd, alpha=0.15)
    ax1.plot(fractions_hub, edges_hub, "s--", linewidth=2.0, markersize=6, color=color_hub, label="Targeted Hub")
    ax1.set_title("A. Synaptic Edges Remaining", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Remaining Edges (% of Baseline)", fontsize=10, fontweight="bold")
    ax1.set_xticks([0, 1, 5, 10, 20])
    ax1.set_xlim(-0.5, 21.0)
    ax1.set_ylim(bottom=0, top=105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    # Panel B: Remaining Synaptic Weight
    ax2.errorbar(
        fractions_rnd,
        weights_rnd,
        yerr=weights_rnd_std,
        fmt="o-",
        linewidth=2.0,
        markersize=6,
        color=color_rnd,
        capsize=4,
        label="Random (Mean ± SD)",
    )
    ax2.fill_between(fractions_rnd, weights_rnd - weights_rnd_std, weights_rnd + weights_rnd_std, color=color_rnd, alpha=0.15)
    ax2.plot(fractions_hub, weights_hub, "s--", linewidth=2.0, markersize=6, color=color_hub, label="Targeted Hub")
    ax2.set_title("B. Total Synaptic Weight Remaining", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Remaining Synaptic Weight (% of Baseline)", fontsize=10, fontweight="bold")
    ax2.set_xticks([0, 1, 5, 10, 20])
    ax2.set_xlim(-0.5, 21.0)
    ax2.set_ylim(bottom=0, top=105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    plt.suptitle("Structural Connectivity Degradation Under Virtual Lesions", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig4_path = figures_dir / "lesion_network_connectivity.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 4 to: {fig4_path}")

    # =========================================================================
    # Figure 5: Comprehensive Summary Dashboard
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel (0, 0): Relative Change in Firing Rate (%)
    ax = axes[0, 0]
    rel_rate_rnd = np.array([0.0] + list(rnd_df["relative_change_rate_mean"].values * 100.0))
    rel_rate_rnd_std = np.array([0.0] + list(rnd_df["relative_change_rate_std"].values * 100.0))
    rel_rate_hub = np.array([0.0] + list(hub_df["relative_change_rate_mean"].values * 100.0))

    ax.errorbar(
        fractions_rnd,
        rel_rate_rnd,
        yerr=rel_rate_rnd_std,
        fmt="o-",
        linewidth=2.0,
        markersize=6,
        color=color_rnd,
        capsize=4,
        label="Random (Mean ± SD)",
    )
    ax.fill_between(fractions_rnd, rel_rate_rnd - rel_rate_rnd_std, rel_rate_rnd + rel_rate_rnd_std, color=color_rnd, alpha=0.15)
    ax.plot(fractions_hub, rel_rate_hub, "s--", linewidth=2.0, markersize=6, color=color_hub, label="Targeted Hub")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.2)
    ax.set_title("A. Relative Change in Mean Firing Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Δ Firing Rate (%)", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1, 5, 10, 20])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    # Panel (0, 1): Activity Robustness vs Network Robustness
    ax = axes[0, 1]
    net_rob_rnd = np.array([1.0] + list(rnd_df["network_robustness_mean"].values))
    act_rob_rnd = np.array([1.0] + list(rnd_df["activity_robustness_mean"].values))
    net_rob_hub = np.array([1.0] + list(hub_df["network_robustness_mean"].values))
    act_rob_hub = np.array([1.0] + list(hub_df["activity_robustness_mean"].values))

    ax.plot(net_rob_rnd, act_rob_rnd, "o-", linewidth=2.0, markersize=7, color=color_rnd, label="Random Lesion Trajectory")
    for i, frac in enumerate([0, 1, 5, 10, 20]):
        ax.annotate(f"{frac}%", (net_rob_rnd[i], act_rob_rnd[i]), textcoords="offset points", xytext=(5, -5), fontsize=8)

    ax.plot(net_rob_hub, act_rob_hub, "s--", linewidth=2.0, markersize=7, color=color_hub, label="Hub Lesion Trajectory")
    for i, frac in enumerate([0, 1, 5, 10, 20]):
        ax.annotate(f"{frac}%", (net_rob_hub[i], act_rob_hub[i]), textcoords="offset points", xytext=(-18, 5), fontsize=8)

    # Diagonal reference line (proportional activity loss)
    ax.plot([0, 1.05], [0, 1.05], ":", color="gray", alpha=0.7, label="Proportional Robustness (1:1)")
    ax.set_title("B. Activity Robustness vs Network Robustness", fontsize=11, fontweight="bold")
    ax.set_xlabel("Network Robustness (Remaining Edges / Baseline Edges)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Activity Robustness (Lesion Rate / Baseline Rate)", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8F9FA", fontsize=9, loc="upper left")

    # Panel (1, 0): Weakly Connected Component Fraction
    ax = axes[1, 0]
    wcc_rnd = np.array([1.0] + list(rnd_df["largest_wcc_relative_fraction_mean"].values)) * 100.0
    wcc_hub = np.array([1.0] + list(hub_df["largest_wcc_relative_fraction_mean"].values)) * 100.0

    ax.plot(fractions_rnd, wcc_rnd, "o-", linewidth=2.0, markersize=6, color=color_rnd, label="Random Lesion")
    ax.plot(fractions_hub, wcc_hub, "s--", linewidth=2.0, markersize=6, color=color_hub, label="Targeted Hub")
    ax.set_title("C. Giant Component Integrity (WCC)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Largest WCC (% of Baseline)", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1, 5, 10, 20])
    ax.set_xlim(-0.5, 21.0)
    ax.set_ylim(bottom=0, top=105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    # Panel (1, 1): Bar comparison at 10% and 20% Lesion
    ax = axes[1, 1]
    labels = ["10% Lesion", "20% Lesion"]
    x = np.arange(len(labels))
    width = 0.35

    hub_10_20 = [hub_df[hub_df["lesion_fraction"] == 0.10]["mean_firing_rate_mean"].iloc[0],
                 hub_df[hub_df["lesion_fraction"] == 0.20]["mean_firing_rate_mean"].iloc[0]]
    rnd_10_20 = [rnd_df[rnd_df["lesion_fraction"] == 0.10]["mean_firing_rate_mean"].iloc[0],
                 rnd_df[rnd_df["lesion_fraction"] == 0.20]["mean_firing_rate_mean"].iloc[0]]
    rnd_std_10_20 = [rnd_df[rnd_df["lesion_fraction"] == 0.10]["mean_firing_rate_std"].iloc[0],
                     rnd_df[rnd_df["lesion_fraction"] == 0.20]["mean_firing_rate_std"].iloc[0]]

    rects1 = ax.bar(x - width / 2, rnd_10_20, width, yerr=rnd_std_10_20, capsize=4, label="Random Lesion", color=color_rnd)
    rects2 = ax.bar(x + width / 2, hub_10_20, width, label="Targeted Hub Lesion", color=color_hub)

    ax.axhline(base_rate, color=color_base, linestyle=":", linewidth=1.5, label=f"Baseline ({base_rate:.2f} Hz)")
    ax.set_title("D. Firing Rate Degradation at 10% vs 20% Ablation", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Firing Rate (Hz)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    plt.suptitle("Virtual Lesion Analysis Summary: Connectivity vs Activity Dynamics", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig5_path = figures_dir / "lesion_effect_summary.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 5 to: {fig5_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Virtual Lesion Analysis on Real 1,000-Neuron Drosophila Connectome Subnetwork."
    )
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed parquet data.")
    parser.add_argument("--neurons", type=int, default=1000, help="Subnetwork neuron count (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for subnetwork extraction and baseline LIF run.")
    parser.add_argument("--duration", type=float, default=1000.0, help="Simulation duration in ms (default: 1000.0).")
    parser.add_argument("--dt", type=float, default=0.5, help="Euler timestep in ms (default: 0.5).")
    parser.add_argument("--weight-scale", type=float, default=0.01, help="Synaptic coupling scale alpha (default: 0.01).")
    parser.add_argument("--external-current", type=float, default=18.0, help="External current amplitude (default: 18.0).")
    parser.add_argument("--stimulus", type=str, default="pulse", help="Stimulus mode (default: pulse).")
    parser.add_argument("--pulse-start", type=float, default=200.0, help="Pulse onset in ms (default: 200.0).")
    parser.add_argument("--pulse-end", type=float, default=600.0, help="Pulse offset in ms (default: 600.0).")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="Output directory for CSV tables.")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="Output directory for figures.")
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" VIRTUAL LESION EXPERIMENT RUNNER (PHASE 4)")
    print("=" * 70)
    print(f" Connectome Subnetwork:     {args.neurons:,} neurons (highest_degree)")
    print(f" Fixed Baseline Parameters: tau_m=20ms, V_rest=-65mV, V_reset=-70mV, V_th=-50mV")
    print(f" Numerical Step & Duration: dt={args.dt}ms, duration={args.duration}ms")
    print(f" Synaptic Weight Scale:     alpha={args.weight_scale} (computational coupling scale)")
    print(f" Stimulus Protocol:         mode={args.stimulus} (I={args.external_current}, window={args.pulse_start}-{args.pulse_end}ms)")
    print(f" Baseline Random Seed:      {args.seed}")
    print("=" * 70 + "\n")

    # 1. Load Real 1,000-Neuron Subnetwork (Janelia FlyEM Hemibrain v1.2.1)
    logger.info(f"Extracting {args.neurons}-neuron subnetwork from {args.processed_dir}...")
    baseline_graph, diag = extract_real_subnetwork(
        processed_dir=args.processed_dir,
        max_neurons=args.neurons,
        strategy="highest_degree",
        seed=args.seed,
        verbose=False,
    )

    base_n = baseline_graph.number_of_nodes()
    base_e = baseline_graph.number_of_edges()
    base_weight = sum(d.get("weight", 1.0) for _, _, d in baseline_graph.edges(data=True))
    base_wcc = max(len(c) for c in nx.weakly_connected_components(baseline_graph))

    logger.info(
        f"Baseline Graph Loaded: {base_n} neurons, {base_e} directed edges, "
        f"{base_weight:.0f} total synaptic weight, largest WCC = {base_wcc}"
    )

    # Configure Fixed LIF Parameters (identical across all experimental runs)
    fixed_lif_config = LIFConfig(
        tau_m=20.0,
        v_rest=-65.0,
        v_reset=-70.0,
        v_threshold=-50.0,
        t_ref=2.0,
        dt=args.dt,
        duration=args.duration,
        synaptic_weight_scale=args.weight_scale,
        external_stimulus_mode=args.stimulus,
        external_current=args.external_current,
        pulse_start=args.pulse_start,
        pulse_end=args.pulse_end,
        stimulus_fraction=0.25,
        random_seed=args.seed,  # Fixed seed for deterministic LIF stimulus & noise
    )

    raw_results: List[Dict[str, Any]] = []
    graph_metrics_results: List[Dict[str, Any]] = []

    tracemalloc.start()
    exp_t0 = time.perf_counter()

    # =========================================================================
    # CONDITION A: BASELINE (No Lesion)
    # =========================================================================
    logger.info("Executing Condition A: Baseline Intact Simulation...")
    base_raw, base_graph_m, base_sim_res = run_single_condition(
        network_graph=baseline_graph.copy(),
        config=fixed_lif_config,
        strategy="baseline",
        fraction=0.0,
        seed=args.seed,
        baseline_metrics=None,
        ablated_nodes=[],
    )
    raw_results.append(base_raw)
    graph_metrics_results.append(base_graph_m)

    baseline_ref = {
        "mean_firing_rate_hz": base_raw["mean_firing_rate_hz"],
        "total_spikes": base_raw["total_spikes"],
        "baseline_edges": base_e,
        "baseline_weight": base_weight,
        "baseline_wcc": base_wcc,
        "baseline_neurons": base_n,
    }

    logger.info(
        f"Baseline Completed: Spikes = {base_raw['total_spikes']:,} | "
        f"Mean Rate = {base_raw['mean_firing_rate_hz']:.2f} Hz | "
        f"Active = {base_raw['active_neuron_fraction']*100:.1f}% | "
        f"Stable = {base_raw['simulation_stable']} ({base_raw['runtime_seconds']:.2f}s)"
    )

    # Experimental fractions and random seeds
    lesion_fractions = [0.01, 0.05, 0.10, 0.20]
    random_seeds = [42, 123, 456, 789, 1000]

    # =========================================================================
    # CONDITION B: RANDOM LESIONS (4 fractions x 5 seeds = 20 runs)
    # =========================================================================
    logger.info("\nExecuting Condition B: Random Lesions (4 fractions x 5 seeds = 20 runs)...")
    for frac in lesion_fractions:
        for r_seed in random_seeds:
            # 1. Lesion occurs BEFORE simulation on an isolated copy
            lesioned_g, ablated_ids = apply_random_lesion(baseline_graph, lesion_fraction=frac, seed=r_seed)

            # Invariant checks
            assert baseline_graph.number_of_nodes() == base_n, "Mutation detected on baseline graph!"
            assert baseline_graph.number_of_edges() == base_e, "Mutation detected on baseline graph!"
            assert lesioned_g.number_of_nodes() == base_n - len(ablated_ids)

            # 2. Run simulation on lesioned graph
            r_raw, r_graph_m, _ = run_single_condition(
                network_graph=lesioned_g,
                config=fixed_lif_config,
                strategy="random",
                fraction=frac,
                seed=r_seed,
                baseline_metrics=baseline_ref,
                ablated_nodes=ablated_ids,
            )
            raw_results.append(r_raw)
            graph_metrics_results.append(r_graph_m)

            logger.info(
                f"[Random] Frac: {frac*100:>2.0f}% | Seed: {r_seed:>4d} | Removed: {len(ablated_ids):>3d} | "
                f"Rate: {r_raw['mean_firing_rate_hz']:>5.2f} Hz (Δ: {r_raw['relative_change_from_baseline']*100:>+5.1f}%) | "
                f"Edges: {r_raw['remaining_edges']:>6d} ({r_raw['network_robustness']*100:>4.1f}%)"
            )

    # =========================================================================
    # CONDITION C: TARGETED HUB LESIONS (4 fractions = 4 runs)
    # =========================================================================
    logger.info("\nExecuting Condition C: Targeted Hub Lesions (Highest Pre-Lesion Degree)...")
    for frac in lesion_fractions:
        # 1. Hub lesion calculated on original pre-lesion degree with deterministic tie-breaking
        lesioned_g, ablated_ids = apply_hub_lesion(baseline_graph, lesion_fraction=frac, weighted=False)

        # Invariant checks
        assert baseline_graph.number_of_nodes() == base_n, "Mutation detected on baseline graph!"
        assert baseline_graph.number_of_edges() == base_e, "Mutation detected on baseline graph!"
        assert lesioned_g.number_of_nodes() == base_n - len(ablated_ids)

        # 2. Run simulation on lesioned graph
        h_raw, h_graph_m, _ = run_single_condition(
            network_graph=lesioned_g,
            config=fixed_lif_config,
            strategy="hub",
            fraction=frac,
            seed=None,
            baseline_metrics=baseline_ref,
            ablated_nodes=ablated_ids,
        )
        raw_results.append(h_raw)
        graph_metrics_results.append(h_graph_m)

        logger.info(
            f"[Hub   ] Frac: {frac*100:>2.0f}% | Removed: {len(ablated_ids):>3d} | "
            f"Rate: {h_raw['mean_firing_rate_hz']:>5.2f} Hz (Δ: {h_raw['relative_change_from_baseline']*100:>+5.1f}%) | "
            f"Edges: {h_raw['remaining_edges']:>6d} ({h_raw['network_robustness']*100:>4.1f}%)"
        )

    exp_runtime = time.perf_counter() - exp_t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_mem / (1024 * 1024)

    # Verify baseline graph is STILL intact at the end
    assert baseline_graph.number_of_nodes() == base_n, "Baseline graph node count corrupted!"
    assert baseline_graph.number_of_edges() == base_e, "Baseline graph edge count corrupted!"

    # =========================================================================
    # EXPORT CSV TABLES
    # =========================================================================
    raw_df = pd.DataFrame(raw_results)
    raw_csv_path = tables_dir / "lesion_raw_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    logger.info(f"\nExported raw results table (25 runs) to: {raw_csv_path}")

    graph_df = pd.DataFrame(graph_metrics_results)
    graph_csv_path = tables_dir / "lesion_graph_metrics.csv"
    graph_df.to_csv(graph_csv_path, index=False)
    logger.info(f"Exported graph metrics table to: {graph_csv_path}")

    summary_df = generate_summary_table(raw_df)
    summary_csv_path = tables_dir / "lesion_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Exported summary statistics table to: {summary_csv_path}")

    # =========================================================================
    # GENERATE PUBLICATION FIGURES
    # =========================================================================
    logger.info("Generating publication figures...")
    generate_lesion_figures(summary_df, raw_df, figures_dir)

    # =========================================================================
    # PRINT EXPERIMENTAL REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print(" VIRTUAL LESION EXPERIMENT SUMMARY REPORT")
    print("=" * 70)
    print(f" Total Experimental Runs:      {len(raw_df)} (1 Baseline + 20 Random + 4 Hub)")
    print(f" Total Execution Time:         {exp_runtime:.2f} seconds")
    print(f" Peak Memory Usage:            {peak_ram_mb:.2f} MB")
    print("-" * 70)
    print(" BASELINE REFERENCE (Intact Subnetwork):")
    print(f"   Neurons: {base_n} | Edges: {base_e:,} | Weight: {base_weight:,.0f} | WCC: {base_wcc}")
    print(f"   Mean Rate: {base_raw['mean_firing_rate_hz']:.2f} Hz | Total Spikes: {base_raw['total_spikes']:,} | Active: {base_raw['active_neuron_fraction']*100:.1f}%")
    print("-" * 70)
    print(" SUMMARY BY CONDITION:")
    print(f"{'Condition':<10} {'Frac':<6} {'Removed':<8} {'Mean Rate (Hz)':<18} {'Total Spikes':<16} {'Rem Edges %':<12} {'Act Robustness':<14}")
    print("-" * 88)
    for _, row in summary_df.iterrows():
        strat = row["lesion_strategy"]
        f_pct = f"{row['lesion_fraction']*100:.0f}%"
        rem = int(row["lesion_count"])
        if strat == "random":
            rate_str = f"{row['mean_firing_rate_mean']:.2f} ± {row['mean_firing_rate_std']:.2f}"
            spk_str = f"{row['total_spikes_mean']:.0f} ± {row['total_spikes_std']:.0f}"
            edges_str = f"{row['network_robustness_mean']*100:.1f}%"
            act_rob_str = f"{row['activity_robustness_mean']:.3f} ± {row['activity_robustness_std']:.3f}"
        else:
            rate_str = f"{row['mean_firing_rate_mean']:.2f}"
            spk_str = f"{row['total_spikes_mean']:.0f}"
            edges_str = f"{row['network_robustness_mean']*100:.1f}%"
            act_rob_str = f"{row['activity_robustness_mean']:.3f}"

        print(f"{strat:<10} {f_pct:<6} {rem:<8} {rate_str:<18} {spk_str:<16} {edges_str:<12} {act_rob_str:<14}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
