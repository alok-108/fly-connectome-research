"""
Experiment 04: Stimulus-Dependence of Lesion Robustness (Phase 5).

Investigates whether network activity robustness to neuronal lesions (random vs hub)
persists when external stimulation is progressively reduced:
- Stimulus fractions: 25%, 10%, 5%, 1%, 0%
- Lesion fractions: 0% (intact baseline), 5%, 10%, 20%
- Lesion strategies: Intact (5 runs), Hub (15 runs), Random (75 runs across 5 seeds: 42, 123, 456, 789, 1000)
- Total simulations: 95 runs.

All experimental conditions normalize against their OWN intact baseline (0% lesion).
Direct stimulation is selected on the intact network before lesioning and naturally decays
as stimulated neurons are ablated.
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
from src.lesion import apply_hub_lesion, apply_random_lesion
from src.simulation import (
    LIFConfig,
    LIFNetwork,
    SimulationResult,
    compute_temporal_metrics,
    select_stimulated_neurons,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_single_simulation(
    graph: nx.DiGraph,
    config: LIFConfig,
    stimulated_neuron_ids: List[int],
) -> Tuple[SimulationResult, Dict[str, Any], Dict[str, Any], float]:
    """
    Executes a single LIF simulation and extracts activity & temporal metrics.

    Returns:
        (sim_res, activity_metrics, temporal_metrics, runtime_sec)
    """
    network = LIFNetwork(graph)
    t0 = time.perf_counter()
    sim_res = network.simulate(config, record_samples=0, stimulated_neuron_ids=stimulated_neuron_ids)
    dur = time.perf_counter() - t0

    n_surviving = graph.number_of_nodes()
    m = sim_res.metrics

    # Check for NaN / Inf in outputs
    rates = sim_res.firing_rates
    num_nan_inf = int(np.isnan(rates).sum() + np.isinf(rates).sum())

    pop_rates = sim_res.population_rate
    pop_peak = float(np.max(pop_rates)) if len(pop_rates) > 0 else 0.0

    activity_m = {
        "total_spikes": int(m["total_spikes"]),
        "mean_firing_rate": float(m["mean_firing_rate_hz"]),
        "median_firing_rate": float(m["median_firing_rate_hz"]),
        "active_neuron_fraction": float(m["active_neuron_fraction"]),
        "max_firing_rate": float(m["max_firing_rate_hz"]),
        "peak_population_rate": round(pop_peak, 4),
        "num_nan_inf_values": num_nan_inf,
        "simulation_stable": sim_res.is_stable,
    }

    temporal_m = compute_temporal_metrics(
        spike_times=sim_res.spike_times,
        num_neurons=n_surviving,
        stimulus_window=(config.pulse_start, config.pulse_end),
        post_window=(config.pulse_end, config.duration),
    )

    return sim_res, activity_m, temporal_m, dur


def generate_phase5_figures(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Renders 8 publication-quality scientific figures for Phase 5.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("tableau-colorblind10")

    stim_fractions = [0.25, 0.10, 0.05, 0.01, 0.00]
    colors = {
        0.25: "#1f77b4",  # Blue
        0.10: "#ff7f0e",  # Orange
        0.05: "#2ca02c",  # Green
        0.01: "#d62728",  # Red
        0.00: "#7f7f7f",  # Gray
    }
    stim_labels = {
        0.25: "25% Stim (250 neurons)",
        0.10: "10% Stim (100 neurons)",
        0.05: "5% Stim (50 neurons)",
        0.01: "1% Stim (10 neurons)",
        0.00: "0% Stim (0 neurons)",
    }

    lesion_pcts = [0, 5, 10, 20]

    # Helper to get curves for a given strategy and stimulus fraction
    def get_curve(df: pd.DataFrame, strat: str, stim_f: float, col: str):
        sub = df[(df["lesion_strategy"] == strat) & (np.isclose(df["stimulus_fraction"], stim_f))].sort_values("lesion_fraction")
        # Ensure 0% lesion is included from intact
        intact_sub = df[(df["lesion_strategy"] == "intact") & (np.isclose(df["stimulus_fraction"], stim_f))]
        intact_val = intact_sub[col].iloc[0] if not intact_sub.empty else 0.0

        les_vals = [intact_val]
        les_x = [0.0]
        for lp in [0.05, 0.10, 0.20]:
            r = sub[np.isclose(sub["lesion_fraction"], lp)]
            if not r.empty:
                les_vals.append(r[col].iloc[0])
                les_x.append(lp * 100.0)
        return np.array(les_x), np.array(les_vals)

    def get_random_err_curve(df: pd.DataFrame, stim_f: float, mean_col: str, std_col: str):
        sub = df[(df["lesion_strategy"] == "random") & (np.isclose(df["stimulus_fraction"], stim_f))].sort_values("lesion_fraction")
        intact_sub = df[(df["lesion_strategy"] == "intact") & (np.isclose(df["stimulus_fraction"], stim_f))]
        intact_val = intact_sub[mean_col].iloc[0] if not intact_sub.empty else 0.0

        les_x = [0.0]
        means = [intact_val]
        stds = [0.0]
        for lp in [0.05, 0.10, 0.20]:
            r = sub[np.isclose(sub["lesion_fraction"], lp)]
            if not r.empty:
                means.append(r[mean_col].iloc[0])
                stds.append(r[std_col].iloc[0])
                les_x.append(lp * 100.0)
        return np.array(les_x), np.array(means), np.array(stds)

    # =========================================================================
    # Figure 1: Mean Firing Rate vs Lesion Percentage (2 Panels: Random vs Hub)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for sf in stim_fractions:
        # Random
        x, y, err = get_random_err_curve(summary_df, sf, "mean_firing_rate_mean", "mean_firing_rate_std")
        ax1.errorbar(x, y, yerr=err, fmt="o-", linewidth=1.8, markersize=5, color=colors[sf], capsize=3, label=stim_labels[sf])
        ax1.fill_between(x, y - err, y + err, color=colors[sf], alpha=0.1)

        # Hub
        xh, yh = get_curve(summary_df, "hub", sf, "mean_firing_rate_mean")
        ax2.plot(xh, yh, "s--", linewidth=1.8, markersize=5, color=colors[sf], label=stim_labels[sf])

    ax1.set_title("A. Random Lesions (Mean ± SD, 5 seeds)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean Firing Rate (Hz)", fontsize=10, fontweight="bold")
    ax1.set_xticks(lesion_pcts)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    ax2.set_title("B. Targeted Hub Lesions", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Firing Rate (Hz)", fontsize=10, fontweight="bold")
    ax2.set_xticks(lesion_pcts)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    plt.suptitle("Population Mean Firing Rate Across Stimulus and Lesion Conditions", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig1_path = figures_dir / "stimulus_mean_firing_rate.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 1 to: {fig1_path}")

    # =========================================================================
    # Figure 2: Activity Robustness vs Lesion Percentage (Normalized to own baseline)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Exclude 0% stimulus from robustness plot if baseline is 0 (NaN)
    stim_fractions_active = [0.25, 0.10, 0.05, 0.01]

    for sf in stim_fractions_active:
        x, y, err = get_random_err_curve(summary_df, sf, "activity_robustness_mean", "activity_robustness_std")
        ax1.errorbar(x, y, yerr=err, fmt="o-", linewidth=1.8, markersize=5, color=colors[sf], capsize=3, label=stim_labels[sf])
        ax1.fill_between(x, y - err, y + err, color=colors[sf], alpha=0.1)

        xh, yh = get_curve(summary_df, "hub", sf, "activity_robustness_mean")
        ax2.plot(xh, yh, "s--", linewidth=1.8, markersize=5, color=colors[sf], label=stim_labels[sf])

    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="Intact Baseline (1.0)")
    ax1.set_title("A. Random Lesion Activity Robustness", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Activity Robustness (Lesion Rate / Intact Rate)", fontsize=10, fontweight="bold")
    ax1.set_xticks(lesion_pcts)
    ax1.set_ylim(0.8, 1.25)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="Intact Baseline (1.0)")
    ax2.set_title("B. Targeted Hub Lesion Activity Robustness", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Activity Robustness (Lesion Rate / Intact Rate)", fontsize=10, fontweight="bold")
    ax2.set_xticks(lesion_pcts)
    ax2.set_ylim(0.8, 1.25)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    plt.suptitle("Activity Robustness Normalized to Stimulus-Specific Intact Baselines", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig2_path = figures_dir / "stimulus_activity_robustness.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 2 to: {fig2_path}")

    # =========================================================================
    # Figure 3: Total Spike Count vs Lesion Percentage
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for sf in stim_fractions:
        x, y, err = get_random_err_curve(summary_df, sf, "spike_count_mean", "spike_count_std")
        ax1.errorbar(x, y, yerr=err, fmt="o-", linewidth=1.8, markersize=5, color=colors[sf], capsize=3, label=stim_labels[sf])
        ax1.fill_between(x, y - err, y + err, color=colors[sf], alpha=0.1)

        xh, yh = get_curve(summary_df, "hub", sf, "spike_count_mean")
        ax2.plot(xh, yh, "s--", linewidth=1.8, markersize=5, color=colors[sf], label=stim_labels[sf])

    ax1.set_title("A. Random Lesions (Total Spikes)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Total Spikes (1000 ms)", fontsize=10, fontweight="bold")
    ax1.set_xticks(lesion_pcts)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    ax2.set_title("B. Targeted Hub Lesions (Total Spikes)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Total Spikes (1000 ms)", fontsize=10, fontweight="bold")
    ax2.set_xticks(lesion_pcts)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    plt.suptitle("Total Population Spike Count Degradation Across External Drive Levels", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig3_path = figures_dir / "stimulus_spike_count.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 3 to: {fig3_path}")

    # =========================================================================
    # Figure 4: Active Neuron Fraction vs Lesion Percentage
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for sf in stim_fractions:
        x, y, err = get_random_err_curve(summary_df, sf, "active_fraction_mean", "active_fraction_std")
        y_pct = y * 100.0
        err_pct = err * 100.0
        ax1.errorbar(x, y_pct, yerr=err_pct, fmt="o-", linewidth=1.8, markersize=5, color=colors[sf], capsize=3, label=stim_labels[sf])
        ax1.fill_between(x, y_pct - err_pct, y_pct + err_pct, color=colors[sf], alpha=0.1)

        xh, yh = get_curve(summary_df, "hub", sf, "active_fraction_mean")
        ax2.plot(xh, yh * 100.0, "s--", linewidth=1.8, markersize=5, color=colors[sf], label=stim_labels[sf])

    ax1.set_title("A. Random Lesions (Active Fraction)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Active Neurons (% Firing ≥1 Spike)", fontsize=10, fontweight="bold")
    ax1.set_xticks(lesion_pcts)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    ax2.set_title("B. Targeted Hub Lesions (Active Fraction)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Active Neurons (% Firing ≥1 Spike)", fontsize=10, fontweight="bold")
    ax2.set_xticks(lesion_pcts)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#F8F9FA", fontsize=9)

    plt.suptitle("Active Neuron Fraction (% of Surviving Neurons Emitting Spikes)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig4_path = figures_dir / "stimulus_active_fraction.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 4 to: {fig4_path}")

    # =========================================================================
    # Figure 5: Stimulus-Window vs Post-Stimulus Firing Rate
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for sf in [0.25, 0.10, 0.05, 0.01]:
        # Panel 1: Stimulus Window (200-600 ms)
        x_rnd, y_rnd_stim, _ = get_random_err_curve(summary_df, sf, "stimulus_window_rate_mean", "mean_firing_rate_std")
        xh, yh_stim = get_curve(summary_df, "hub", sf, "stimulus_window_rate_mean")
        ax1.plot(x_rnd, y_rnd_stim, "o-", linewidth=1.8, color=colors[sf], label=f"{stim_labels[sf]} (Random)")
        ax1.plot(xh, yh_stim, "s--", linewidth=1.5, alpha=0.8, color=colors[sf], label=f"{stim_labels[sf]} (Hub)")

        # Panel 2: Post-Stimulus Window (600-1000 ms)
        x_rnd_p, y_rnd_post, _ = get_random_err_curve(summary_df, sf, "post_stimulus_rate_mean", "mean_firing_rate_std")
        xh_p, yh_post = get_curve(summary_df, "hub", sf, "post_stimulus_rate_mean")
        ax2.plot(x_rnd_p, y_rnd_post, "o-", linewidth=1.8, color=colors[sf], label=f"{stim_labels[sf]} (Random)")
        ax2.plot(xh_p, yh_post, "s--", linewidth=1.5, alpha=0.8, color=colors[sf], label=f"{stim_labels[sf]} (Hub)")

    ax1.set_title("A. Stimulus Window (200–600 ms)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean Rate in Window (Hz)", fontsize=10, fontweight="bold")
    ax1.set_xticks(lesion_pcts)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#F8F9FA", fontsize=8)

    ax2.set_title("B. Post-Stimulus Window (600–1000 ms)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Rate in Window (Hz)", fontsize=10, fontweight="bold")
    ax2.set_xticks(lesion_pcts)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#F8F9FA", fontsize=8)

    plt.suptitle("Temporal Decomposition: Stimulus-Driven vs Post-Stimulus Self-Sustained Dynamics", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig5_path = figures_dir / "stimulus_window_vs_post_rate.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 5 to: {fig5_path}")

    # =========================================================================
    # Figure 6: Edge Retention vs Activity Robustness
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for sf in [0.25, 0.10, 0.05, 0.01]:
        # Random points
        sub_rnd = summary_df[(summary_df["lesion_strategy"] == "random") & (np.isclose(summary_df["stimulus_fraction"], sf))]
        sub_intact = summary_df[(summary_df["lesion_strategy"] == "intact") & (np.isclose(summary_df["stimulus_fraction"], sf))]

        edges_rnd = [1.0] + list(sub_rnd["edge_retention_mean"].values)
        rob_rnd = [1.0] + list(sub_rnd["activity_robustness_mean"].values)
        ax.plot(edges_rnd, rob_rnd, "o-", linewidth=2.0, markersize=6, color=colors[sf], label=f"{stim_labels[sf]} (Random)")

        # Hub points
        sub_hub = summary_df[(summary_df["lesion_strategy"] == "hub") & (np.isclose(summary_df["stimulus_fraction"], sf))]
        edges_hub = [1.0] + list(sub_hub["edge_retention_mean"].values)
        rob_hub = [1.0] + list(sub_hub["activity_robustness_mean"].values)
        ax.plot(edges_hub, rob_hub, "s--", linewidth=1.8, markersize=6, color=colors[sf], alpha=0.8, label=f"{stim_labels[sf]} (Hub)")

    ax.plot([0.35, 1.05], [0.35, 1.05], ":", color="gray", alpha=0.7, label="Proportional Structural Loss (1:1)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title("Structural Edge Retention vs Functional Activity Robustness", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Edge Retention (Remaining Edges / Baseline Edges)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Activity Robustness (Lesion Rate / Intact Rate)", fontsize=11, fontweight="bold")
    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(0.8, 1.25)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8F9FA", fontsize=8.5, loc="upper left")

    plt.tight_layout()
    fig6_path = figures_dir / "stimulus_edge_retention_vs_robustness.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 6 to: {fig6_path}")

    # =========================================================================
    # Figure 7: Effective Stimulus Fraction After Lesions
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for sf in [0.25, 0.10, 0.05, 0.01]:
        # Random surviving stimulated count
        sub_rnd = raw_df[(raw_df["lesion_strategy"] == "random") & (np.isclose(raw_df["stimulus_fraction"], sf))]
        # Mean surviving count per lesion fraction
        surv_rnd = [sf * 1000.0]
        eff_rnd = [sf * 100.0]
        for lp in [0.05, 0.10, 0.20]:
            r = sub_rnd[np.isclose(sub_rnd["lesion_fraction"], lp)]
            surv_rnd.append(float(r["surviving_stimulated_neurons"].mean()))
            eff_rnd.append(float(r["effective_stimulus_fraction"].mean() * 100.0))

        ax1.plot(lesion_pcts, surv_rnd, "o-", linewidth=1.8, markersize=5, color=colors[sf], label=f"{stim_labels[sf]} (Random)")

        # Hub surviving stimulated count
        sub_hub = raw_df[(raw_df["lesion_strategy"] == "hub") & (np.isclose(raw_df["stimulus_fraction"], sf))]
        surv_hub = [sf * 1000.0]
        eff_hub = [sf * 100.0]
        for lp in [0.05, 0.10, 0.20]:
            r = sub_hub[np.isclose(sub_hub["lesion_fraction"], lp)]
            surv_hub.append(float(r["surviving_stimulated_neurons"].iloc[0]))
            eff_hub.append(float(r["effective_stimulus_fraction"].iloc[0] * 100.0))

        ax1.plot(lesion_pcts, surv_hub, "s--", linewidth=1.8, markersize=5, color=colors[sf], alpha=0.8, label=f"{stim_labels[sf]} (Hub)")

        # Panel 2: Effective fraction of surviving population
        ax2.plot(lesion_pcts, eff_rnd, "o-", linewidth=1.8, markersize=5, color=colors[sf], label=f"{stim_labels[sf]} (Random)")
        ax2.plot(lesion_pcts, eff_hub, "s--", linewidth=1.8, markersize=5, color=colors[sf], alpha=0.8, label=f"{stim_labels[sf]} (Hub)")

    ax1.set_title("A. Surviving Stimulated Neurons (Count)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Surviving Stimulated Neurons", fontsize=10, fontweight="bold")
    ax1.set_xticks(lesion_pcts)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#F8F9FA", fontsize=8.5)

    ax2.set_title("B. Effective Stimulus Fraction of Survivors (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Effective Stimulus Fraction (%)", fontsize=10, fontweight="bold")
    ax2.set_xticks(lesion_pcts)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#F8F9FA", fontsize=8.5)

    plt.suptitle("Survival of Directly Stimulated Neurons Under Lesioning", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig7_path = figures_dir / "stimulus_effective_fraction.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 7 to: {fig7_path}")

    # =========================================================================
    # Figure 8: Heatmaps of Activity Robustness (Rows=Stim Fraction, Cols=Lesion Fraction)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    stim_labels_y = ["25%", "10%", "5%", "1%"]
    les_labels_x = ["0%", "5%", "10%", "20%"]

    matrix_rnd = np.zeros((len(stim_labels_y), len(les_labels_x)))
    matrix_hub = np.zeros((len(stim_labels_y), len(les_labels_x)))

    for i, sf in enumerate([0.25, 0.10, 0.05, 0.01]):
        # 0% intact = 1.0
        matrix_rnd[i, 0] = 1.0
        matrix_hub[i, 0] = 1.0

        for j, lp in enumerate([0.05, 0.10, 0.20], start=1):
            sub_r = summary_df[(summary_df["lesion_strategy"] == "random") & (np.isclose(summary_df["stimulus_fraction"], sf)) & (np.isclose(summary_df["lesion_fraction"], lp))]
            matrix_rnd[i, j] = sub_r["activity_robustness_mean"].iloc[0] if not sub_r.empty else np.nan

            sub_h = summary_df[(summary_df["lesion_strategy"] == "hub") & (np.isclose(summary_df["stimulus_fraction"], sf)) & (np.isclose(summary_df["lesion_fraction"], lp))]
            matrix_hub[i, j] = sub_h["activity_robustness_mean"].iloc[0] if not sub_h.empty else np.nan

    vmin = min(np.nanmin(matrix_rnd), np.nanmin(matrix_hub), 0.90)
    vmax = max(np.nanmax(matrix_rnd), np.nanmax(matrix_hub), 1.15)

    im1 = ax1.imshow(matrix_rnd, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title("A. Random Lesions (Activity Robustness)", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(len(les_labels_x)))
    ax1.set_xticklabels(les_labels_x, fontsize=10, fontweight="bold")
    ax1.set_yticks(range(len(stim_labels_y)))
    ax1.set_yticklabels(stim_labels_y, fontsize=10, fontweight="bold")
    ax1.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Stimulation Fraction (%)", fontsize=10, fontweight="bold")

    for ii in range(len(stim_labels_y)):
        for jj in range(len(les_labels_x)):
            val = matrix_rnd[ii, jj]
            ax1.text(jj, ii, f"{val:.3f}", ha="center", va="center", color="black" if 0.95 <= val <= 1.05 else "white", fontweight="bold", fontsize=9)

    im2 = ax2.imshow(matrix_hub, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title("B. Targeted Hub Lesions (Activity Robustness)", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(les_labels_x)))
    ax2.set_xticklabels(les_labels_x, fontsize=10, fontweight="bold")
    ax2.set_yticks(range(len(stim_labels_y)))
    ax2.set_yticklabels(stim_labels_y, fontsize=10, fontweight="bold")
    ax2.set_xlabel("Lesion Fraction (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Stimulation Fraction (%)", fontsize=10, fontweight="bold")

    for ii in range(len(stim_labels_y)):
        for jj in range(len(les_labels_x)):
            val = matrix_hub[ii, jj]
            ax2.text(jj, ii, f"{val:.3f}", ha="center", va="center", color="black" if 0.95 <= val <= 1.05 else "white", fontweight="bold", fontsize=9)

    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label("Activity Robustness (Lesion Rate / Intact Rate)", fontsize=10, fontweight="bold")

    plt.suptitle("Activity Robustness Heatmap (Stimulus Fraction vs Lesion Fraction)", fontsize=13, fontweight="bold", y=0.98)
    fig8_path = figures_dir / "stimulus_robustness_heatmap.png"
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 8 to: {fig8_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Stimulus-Dependence of Lesion Robustness on Real 1,000-Neuron Drosophila Connectome Subnetwork."
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

    print("\n" + "=" * 75)
    print(" PHASE 5: STIMULUS-DEPENDENCE OF LESION ROBUSTNESS EXPERIMENT")
    print("=" * 75)
    print(f" Connectome Subnetwork:     {args.neurons:,} neurons (highest_degree)")
    print(f" Fixed Baseline Parameters: tau_m=20ms, V_rest=-65mV, V_reset=-70mV, V_th=-50mV")
    print(f" Synaptic Coupling Scale:   alpha={args.weight_scale} (computational coupling scale)")
    print(f" Stimulus Protocol:         mode={args.stimulus} (I={args.external_current}, window={args.pulse_start}-{args.pulse_end}ms)")
    print(" Stimulus Fractions:        25%, 10%, 5%, 1%, 0%")
    print(" Lesion Fractions:          0% (intact), 5%, 10%, 20%")
    print(" Total Planned Simulations: 95 runs (5 Intact + 15 Hub + 75 Random)")
    print("=" * 75 + "\n")

    # 1. Load Real 1,000-Neuron Subnetwork (Janelia FlyEM Hemibrain v1.2.1)
    logger.info(f"Extracting {args.neurons}-neuron subnetwork from {args.processed_dir}...")
    baseline_graph, _ = extract_real_subnetwork(
        processed_dir=args.processed_dir,
        max_neurons=args.neurons,
        strategy="highest_degree",
        seed=args.seed,
        verbose=False,
    )

    base_n = baseline_graph.number_of_nodes()
    base_e = baseline_graph.number_of_edges()
    base_weight = sum(d.get("weight", 1.0) for _, _, d in baseline_graph.edges(data=True))

    logger.info(f"Baseline Graph Loaded: {base_n} neurons, {base_e} edges, {base_weight:.0f} total weight.")

    # Freeze LIFConfig (identical across all experimental runs)
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
        random_seed=args.seed,
    )

    stimulus_fractions = [0.25, 0.10, 0.05, 0.01, 0.00]
    lesion_fractions = [0.05, 0.10, 0.20]
    random_seeds = [42, 123, 456, 789, 1000]

    raw_results: List[Dict[str, Any]] = []
    graph_metrics_results: List[Dict[str, Any]] = []

    # Dictionary to store intact baseline metrics for each stimulus fraction
    intact_baselines: Dict[float, Dict[str, Any]] = {}

    tracemalloc.start()
    t_start_total = time.perf_counter()

    # =========================================================================
    # STEP 1: RUN INTACT BASELINES (5 simulations, 1 per stimulus fraction)
    # =========================================================================
    logger.info("\n--- STEP 1: Running Intact Baselines for all Stimulus Fractions (5 runs) ---")
    stim_sets: Dict[float, List[int]] = {}

    for sf in stimulus_fractions:
        # Pre-select stimulated neurons on intact network
        stim_ids = select_stimulated_neurons(baseline_graph, stimulus_fraction=sf, seed=args.seed)
        stim_sets[sf] = stim_ids

        wcc_comps = list(nx.weakly_connected_components(baseline_graph))
        largest_wcc = max(len(c) for c in wcc_comps) if base_n > 0 else 0

        # Execute simulation
        sim_res, act_m, temp_m, dur = run_single_simulation(
            graph=baseline_graph.copy(),
            config=fixed_lif_config,
            stimulated_neuron_ids=stim_ids,
        )

        intact_baselines[sf] = {
            "mean_firing_rate": act_m["mean_firing_rate"],
            "total_spikes": act_m["total_spikes"],
            "active_neuron_fraction": act_m["active_neuron_fraction"],
            "stimulus_window_rate_hz": temp_m["stimulus_window_rate_hz"],
            "post_stimulus_window_rate_hz": temp_m["post_stimulus_window_rate_hz"],
        }

        # Intact robustness is by definition 1.0 (or NaN if baseline rate is 0)
        is_zero_base = act_m["mean_firing_rate"] == 0.0
        act_rob = np.nan if is_zero_base else 1.0
        spk_rob = np.nan if act_m["total_spikes"] == 0 else 1.0
        act_frac_rob = np.nan if act_m["active_neuron_fraction"] == 0.0 else 1.0
        stim_w_rob = np.nan if temp_m["stimulus_window_rate_hz"] == 0.0 else 1.0
        post_w_rob = np.nan if temp_m["post_stimulus_window_rate_hz"] == 0.0 else 1.0

        raw_row = {
            "stimulus_fraction": sf,
            "initial_stimulated_neurons": len(stim_ids),
            "surviving_stimulated_neurons": len(stim_ids),
            "effective_stimulus_fraction": round(len(stim_ids) / base_n, 4),
            "lesion_strategy": "intact",
            "lesion_fraction": 0.0,
            "random_seed": "NA",
            "initial_neurons": base_n,
            "surviving_neurons": base_n,
            "initial_edges": base_e,
            "surviving_edges": base_e,
            "edge_retention": 1.0,
            "initial_total_synaptic_weight": round(base_weight, 1),
            "surviving_total_synaptic_weight": round(base_weight, 1),
            "synaptic_weight_retention": 1.0,
            "largest_weak_component_size": largest_wcc,
            "largest_wcc_fraction": round(largest_wcc / base_n, 4),
            "total_spikes": act_m["total_spikes"],
            "mean_firing_rate": act_m["mean_firing_rate"],
            "median_firing_rate": act_m["median_firing_rate"],
            "active_neuron_fraction": act_m["active_neuron_fraction"],
            "max_firing_rate": act_m["max_firing_rate"],
            "peak_population_rate": act_m["peak_population_rate"],
            "num_nan_inf_values": act_m["num_nan_inf_values"],
            "simulation_stable": act_m["simulation_stable"],
            "stimulus_window_spikes": temp_m["stimulus_window_spikes"],
            "stimulus_window_rate_hz": temp_m["stimulus_window_rate_hz"],
            "post_stimulus_window_spikes": temp_m["post_stimulus_window_spikes"],
            "post_stimulus_window_rate_hz": temp_m["post_stimulus_window_rate_hz"],
            "stimulus_to_post_ratio": temp_m["stimulus_to_post_ratio"],
            "activity_robustness": act_rob,
            "spike_count_robustness": spk_rob,
            "active_fraction_robustness": act_frac_rob,
            "stimulus_window_rate_robustness": stim_w_rob,
            "post_stimulus_rate_robustness": post_w_rob,
            "baseline_normalization_denominator_rate": act_m["mean_firing_rate"],
            "runtime_seconds": round(dur, 4),
        }
        raw_results.append(raw_row)

        graph_row = {
            "stimulus_fraction": sf,
            "initial_stimulated_neurons": len(stim_ids),
            "surviving_stimulated_neurons": len(stim_ids),
            "effective_stimulus_fraction": round(len(stim_ids) / base_n, 4),
            "lesion_strategy": "intact",
            "lesion_fraction": 0.0,
            "random_seed": "NA",
            "initial_neurons": base_n,
            "surviving_neurons": base_n,
            "initial_edges": base_e,
            "surviving_edges": base_e,
            "edge_retention": 1.0,
            "initial_total_synaptic_weight": round(base_weight, 1),
            "surviving_total_synaptic_weight": round(base_weight, 1),
            "synaptic_weight_retention": 1.0,
            "largest_weak_component_size": largest_wcc,
            "largest_wcc_fraction": round(largest_wcc / base_n, 4),
            "num_weak_components": len(wcc_comps),
        }
        graph_metrics_results.append(graph_row)

        logger.info(
            f"[Intact] Stim: {sf*100:>2.0f}% ({len(stim_ids):>3d} stim) | Rate: {act_m['mean_firing_rate']:>5.2f} Hz | "
            f"Spikes: {act_m['total_spikes']:>5d} | Active: {act_m['active_neuron_fraction']*100:>4.1f}% | "
            f"Stim Window: {temp_m['stimulus_window_rate_hz']:>5.2f} Hz | Post Window: {temp_m['post_stimulus_window_rate_hz']:>5.2f} Hz"
        )

    # =========================================================================
    # STEP 2: RUN HUB LESIONS (15 simulations = 5 stim fractions x 3 lesion levels)
    # =========================================================================
    logger.info("\n--- STEP 2: Running Targeted Hub Lesions (15 runs) ---")
    for sf in stimulus_fractions:
        initial_stim_ids = stim_sets[sf]
        base_ref = intact_baselines[sf]

        for lf in lesion_fractions:
            # Hub lesion derived from intact pre-lesion degree with deterministic tie-breaking
            lesioned_g, ablated_ids = apply_hub_lesion(baseline_graph, lesion_fraction=lf, weighted=False)

            # Invariant check
            assert baseline_graph.number_of_nodes() == base_n, "Baseline graph mutated!"
            assert baseline_graph.number_of_edges() == base_e, "Baseline graph mutated!"

            n_surv = lesioned_g.number_of_nodes()
            e_surv = lesioned_g.number_of_edges()
            w_surv = sum(d.get("weight", 1.0) for _, _, d in lesioned_g.edges(data=True))
            wcc_comps = list(nx.weakly_connected_components(lesioned_g))
            largest_wcc = max(len(c) for c in wcc_comps) if n_surv > 0 else 0

            # Surviving stimulated neurons
            surviving_stim_ids = [nid for nid in initial_stim_ids if nid in lesioned_g]
            n_surv_stim = len(surviving_stim_ids)
            eff_stim_frac = (n_surv_stim / n_surv) if n_surv > 0 else 0.0

            # Run simulation
            sim_res, act_m, temp_m, dur = run_single_simulation(
                graph=lesioned_g,
                config=fixed_lif_config,
                stimulated_neuron_ids=surviving_stim_ids,
            )

            # Robustness normalized against OWN stimulus intact baseline
            base_rate = base_ref["mean_firing_rate"]
            act_rob = (act_m["mean_firing_rate"] / base_rate) if base_rate > 0 else np.nan
            spk_rob = (act_m["total_spikes"] / base_ref["total_spikes"]) if base_ref["total_spikes"] > 0 else np.nan
            act_frac_rob = (act_m["active_neuron_fraction"] / base_ref["active_neuron_fraction"]) if base_ref["active_neuron_fraction"] > 0 else np.nan
            stim_w_rob = (temp_m["stimulus_window_rate_hz"] / base_ref["stimulus_window_rate_hz"]) if base_ref["stimulus_window_rate_hz"] > 0 else np.nan
            post_w_rob = (temp_m["post_stimulus_window_rate_hz"] / base_ref["post_stimulus_window_rate_hz"]) if base_ref["post_stimulus_window_rate_hz"] > 0 else np.nan

            raw_row = {
                "stimulus_fraction": sf,
                "initial_stimulated_neurons": len(initial_stim_ids),
                "surviving_stimulated_neurons": n_surv_stim,
                "effective_stimulus_fraction": round(eff_stim_frac, 4),
                "lesion_strategy": "hub",
                "lesion_fraction": lf,
                "random_seed": "NA",
                "initial_neurons": base_n,
                "surviving_neurons": n_surv,
                "initial_edges": base_e,
                "surviving_edges": e_surv,
                "edge_retention": round(e_surv / base_e, 4),
                "initial_total_synaptic_weight": round(base_weight, 1),
                "surviving_total_synaptic_weight": round(w_surv, 1),
                "synaptic_weight_retention": round(w_surv / base_weight, 4),
                "largest_weak_component_size": largest_wcc,
                "largest_wcc_fraction": round(largest_wcc / n_surv, 4),
                "total_spikes": act_m["total_spikes"],
                "mean_firing_rate": act_m["mean_firing_rate"],
                "median_firing_rate": act_m["median_firing_rate"],
                "active_neuron_fraction": act_m["active_neuron_fraction"],
                "max_firing_rate": act_m["max_firing_rate"],
                "peak_population_rate": act_m["peak_population_rate"],
                "num_nan_inf_values": act_m["num_nan_inf_values"],
                "simulation_stable": act_m["simulation_stable"],
                "stimulus_window_spikes": temp_m["stimulus_window_spikes"],
                "stimulus_window_rate_hz": temp_m["stimulus_window_rate_hz"],
                "post_stimulus_window_spikes": temp_m["post_stimulus_window_spikes"],
                "post_stimulus_window_rate_hz": temp_m["post_stimulus_window_rate_hz"],
                "stimulus_to_post_ratio": temp_m["stimulus_to_post_ratio"],
                "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                "active_fraction_robustness": round(act_frac_rob, 4) if not np.isnan(act_frac_rob) else np.nan,
                "stimulus_window_rate_robustness": round(stim_w_rob, 4) if not np.isnan(stim_w_rob) else np.nan,
                "post_stimulus_rate_robustness": round(post_w_rob, 4) if not np.isnan(post_w_rob) else np.nan,
                "baseline_normalization_denominator_rate": base_rate,
                "runtime_seconds": round(dur, 4),
            }
            raw_results.append(raw_row)

            graph_row = {
                "stimulus_fraction": sf,
                "initial_stimulated_neurons": len(initial_stim_ids),
                "surviving_stimulated_neurons": n_surv_stim,
                "effective_stimulus_fraction": round(eff_stim_frac, 4),
                "lesion_strategy": "hub",
                "lesion_fraction": lf,
                "random_seed": "NA",
                "initial_neurons": base_n,
                "surviving_neurons": n_surv,
                "initial_edges": base_e,
                "surviving_edges": e_surv,
                "edge_retention": round(e_surv / base_e, 4),
                "initial_total_synaptic_weight": round(base_weight, 1),
                "surviving_total_synaptic_weight": round(w_surv, 1),
                "synaptic_weight_retention": round(w_surv / base_weight, 4),
                "largest_weak_component_size": largest_wcc,
                "largest_wcc_fraction": round(largest_wcc / n_surv, 4),
                "num_weak_components": len(wcc_comps),
            }
            graph_metrics_results.append(graph_row)

            rob_str = f"{act_rob:.3f}" if not np.isnan(act_rob) else "NaN"
            logger.info(
                f"[Hub   ] Stim: {sf*100:>2.0f}% | Lesion: {lf*100:>2.0f}% | Surv Stim: {n_surv_stim:>3d} | "
                f"Rate: {act_m['mean_firing_rate']:>5.2f} Hz | ActRob: {rob_str} | Edges: {e_surv:>6d} ({e_surv/base_e*100:>4.1f}%)"
            )

    # =========================================================================
    # STEP 3: RUN RANDOM LESIONS (75 simulations = 5 stim x 3 lesion x 5 seeds)
    # =========================================================================
    logger.info("\n--- STEP 3: Running Random Lesions (75 runs) ---")
    for sf in stimulus_fractions:
        initial_stim_ids = stim_sets[sf]
        base_ref = intact_baselines[sf]

        for lf in lesion_fractions:
            for r_seed in random_seeds:
                lesioned_g, ablated_ids = apply_random_lesion(baseline_graph, lesion_fraction=lf, seed=r_seed)

                assert baseline_graph.number_of_nodes() == base_n, "Baseline graph mutated!"
                assert baseline_graph.number_of_edges() == base_e, "Baseline graph mutated!"

                n_surv = lesioned_g.number_of_nodes()
                e_surv = lesioned_g.number_of_edges()
                w_surv = sum(d.get("weight", 1.0) for _, _, d in lesioned_g.edges(data=True))
                wcc_comps = list(nx.weakly_connected_components(lesioned_g))
                largest_wcc = max(len(c) for c in wcc_comps) if n_surv > 0 else 0

                surviving_stim_ids = [nid for nid in initial_stim_ids if nid in lesioned_g]
                n_surv_stim = len(surviving_stim_ids)
                eff_stim_frac = (n_surv_stim / n_surv) if n_surv > 0 else 0.0

                sim_res, act_m, temp_m, dur = run_single_simulation(
                    graph=lesioned_g,
                    config=fixed_lif_config,
                    stimulated_neuron_ids=surviving_stim_ids,
                )

                base_rate = base_ref["mean_firing_rate"]
                act_rob = (act_m["mean_firing_rate"] / base_rate) if base_rate > 0 else np.nan
                spk_rob = (act_m["total_spikes"] / base_ref["total_spikes"]) if base_ref["total_spikes"] > 0 else np.nan
                act_frac_rob = (act_m["active_neuron_fraction"] / base_ref["active_neuron_fraction"]) if base_ref["active_neuron_fraction"] > 0 else np.nan
                stim_w_rob = (temp_m["stimulus_window_rate_hz"] / base_ref["stimulus_window_rate_hz"]) if base_ref["stimulus_window_rate_hz"] > 0 else np.nan
                post_w_rob = (temp_m["post_stimulus_window_rate_hz"] / base_ref["post_stimulus_window_rate_hz"]) if base_ref["post_stimulus_window_rate_hz"] > 0 else np.nan

                raw_row = {
                    "stimulus_fraction": sf,
                    "initial_stimulated_neurons": len(initial_stim_ids),
                    "surviving_stimulated_neurons": n_surv_stim,
                    "effective_stimulus_fraction": round(eff_stim_frac, 4),
                    "lesion_strategy": "random",
                    "lesion_fraction": lf,
                    "random_seed": r_seed,
                    "initial_neurons": base_n,
                    "surviving_neurons": n_surv,
                    "initial_edges": base_e,
                    "surviving_edges": e_surv,
                    "edge_retention": round(e_surv / base_e, 4),
                    "initial_total_synaptic_weight": round(base_weight, 1),
                    "surviving_total_synaptic_weight": round(w_surv, 1),
                    "synaptic_weight_retention": round(w_surv / base_weight, 4),
                    "largest_weak_component_size": largest_wcc,
                    "largest_wcc_fraction": round(largest_wcc / n_surv, 4),
                    "total_spikes": act_m["total_spikes"],
                    "mean_firing_rate": act_m["mean_firing_rate"],
                    "median_firing_rate": act_m["median_firing_rate"],
                    "active_neuron_fraction": act_m["active_neuron_fraction"],
                    "max_firing_rate": act_m["max_firing_rate"],
                    "peak_population_rate": act_m["peak_population_rate"],
                    "num_nan_inf_values": act_m["num_nan_inf_values"],
                    "simulation_stable": act_m["simulation_stable"],
                    "stimulus_window_spikes": temp_m["stimulus_window_spikes"],
                    "stimulus_window_rate_hz": temp_m["stimulus_window_rate_hz"],
                    "post_stimulus_window_spikes": temp_m["post_stimulus_window_spikes"],
                    "post_stimulus_window_rate_hz": temp_m["post_stimulus_window_rate_hz"],
                    "stimulus_to_post_ratio": temp_m["stimulus_to_post_ratio"],
                    "activity_robustness": round(act_rob, 4) if not np.isnan(act_rob) else np.nan,
                    "spike_count_robustness": round(spk_rob, 4) if not np.isnan(spk_rob) else np.nan,
                    "active_fraction_robustness": round(act_frac_rob, 4) if not np.isnan(act_frac_rob) else np.nan,
                    "stimulus_window_rate_robustness": round(stim_w_rob, 4) if not np.isnan(stim_w_rob) else np.nan,
                    "post_stimulus_rate_robustness": round(post_w_rob, 4) if not np.isnan(post_w_rob) else np.nan,
                    "baseline_normalization_denominator_rate": base_rate,
                    "runtime_seconds": round(dur, 4),
                }
                raw_results.append(raw_row)

                graph_row = {
                    "stimulus_fraction": sf,
                    "initial_stimulated_neurons": len(initial_stim_ids),
                    "surviving_stimulated_neurons": n_surv_stim,
                    "effective_stimulus_fraction": round(eff_stim_frac, 4),
                    "lesion_strategy": "random",
                    "lesion_fraction": lf,
                    "random_seed": r_seed,
                    "initial_neurons": base_n,
                    "surviving_neurons": n_surv,
                    "initial_edges": base_e,
                    "surviving_edges": e_surv,
                    "edge_retention": round(e_surv / base_e, 4),
                    "initial_total_synaptic_weight": round(base_weight, 1),
                    "surviving_total_synaptic_weight": round(w_surv, 1),
                    "synaptic_weight_retention": round(w_surv / base_weight, 4),
                    "largest_weak_component_size": largest_wcc,
                    "largest_wcc_fraction": round(largest_wcc / n_surv, 4),
                    "num_weak_components": len(wcc_comps),
                }
                graph_metrics_results.append(graph_row)

                rob_str = f"{act_rob:.3f}" if not np.isnan(act_rob) else "NaN"
                logger.info(
                    f"[Random] Stim: {sf*100:>2.0f}% | Lesion: {lf*100:>2.0f}% | Seed: {r_seed:>4d} | "
                    f"Rate: {act_m['mean_firing_rate']:>5.2f} Hz | ActRob: {rob_str} | Edges: {e_surv:>6d} ({e_surv/base_e*100:>4.1f}%)"
                )

    total_runtime = time.perf_counter() - t_start_total
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_mem / (1024 * 1024)

    # Invariant verification
    assert baseline_graph.number_of_nodes() == base_n
    assert baseline_graph.number_of_edges() == base_e

    # =========================================================================
    # STEP 4: EXPORT TABLES
    # =========================================================================
    raw_df = pd.DataFrame(raw_results)
    raw_csv = tables_dir / "stimulus_robustness_raw_results.csv"
    raw_df.to_csv(raw_csv, index=False)
    logger.info(f"\nExported raw results table ({len(raw_df)} simulations) to: {raw_csv}")

    graph_df = pd.DataFrame(graph_metrics_results)
    graph_csv = tables_dir / "stimulus_robustness_graph_metrics.csv"
    graph_df.to_csv(graph_csv, index=False)
    logger.info(f"Exported graph metrics table to: {graph_csv}")

    # Build Summary Table
    summary_records = []
    # 1. Intact rows
    for sf in stimulus_fractions:
        sub = raw_df[(raw_df["lesion_strategy"] == "intact") & (np.isclose(raw_df["stimulus_fraction"], sf))].iloc[0]
        summary_records.append({
            "stimulus_fraction": sf,
            "lesion_strategy": "intact",
            "lesion_fraction": 0.0,
            "mean_firing_rate_mean": sub["mean_firing_rate"],
            "mean_firing_rate_std": 0.0,
            "spike_count_mean": sub["total_spikes"],
            "spike_count_std": 0.0,
            "active_fraction_mean": sub["active_neuron_fraction"],
            "active_fraction_std": 0.0,
            "stimulus_window_rate_mean": sub["stimulus_window_rate_hz"],
            "post_stimulus_rate_mean": sub["post_stimulus_window_rate_hz"],
            "activity_robustness_mean": sub["activity_robustness"],
            "activity_robustness_std": 0.0,
            "edge_retention_mean": 1.0,
            "weight_retention_mean": 1.0,
        })

    # 2. Hub rows
    for sf in stimulus_fractions:
        for lf in lesion_fractions:
            sub = raw_df[(raw_df["lesion_strategy"] == "hub") & (np.isclose(raw_df["stimulus_fraction"], sf)) & (np.isclose(raw_df["lesion_fraction"], lf))].iloc[0]
            summary_records.append({
                "stimulus_fraction": sf,
                "lesion_strategy": "hub",
                "lesion_fraction": lf,
                "mean_firing_rate_mean": sub["mean_firing_rate"],
                "mean_firing_rate_std": np.nan,  # Deterministic single run
                "spike_count_mean": sub["total_spikes"],
                "spike_count_std": np.nan,
                "active_fraction_mean": sub["active_neuron_fraction"],
                "active_fraction_std": np.nan,
                "stimulus_window_rate_mean": sub["stimulus_window_rate_hz"],
                "post_stimulus_rate_mean": sub["post_stimulus_window_rate_hz"],
                "activity_robustness_mean": sub["activity_robustness"],
                "activity_robustness_std": np.nan,
                "edge_retention_mean": sub["edge_retention"],
                "weight_retention_mean": sub["synaptic_weight_retention"],
            })

    # 3. Random rows (aggregated over 5 seeds)
    for sf in stimulus_fractions:
        for lf in lesion_fractions:
            sub = raw_df[(raw_df["lesion_strategy"] == "random") & (np.isclose(raw_df["stimulus_fraction"], sf)) & (np.isclose(raw_df["lesion_fraction"], lf))]
            act_rob_series = sub["activity_robustness"].dropna()
            summary_records.append({
                "stimulus_fraction": sf,
                "lesion_strategy": "random",
                "lesion_fraction": lf,
                "mean_firing_rate_mean": round(float(sub["mean_firing_rate"].mean()), 4),
                "mean_firing_rate_std": round(float(sub["mean_firing_rate"].std()), 4),
                "spike_count_mean": round(float(sub["total_spikes"].mean()), 1),
                "spike_count_std": round(float(sub["total_spikes"].std()), 1),
                "active_fraction_mean": round(float(sub["active_neuron_fraction"].mean()), 4),
                "active_fraction_std": round(float(sub["active_neuron_fraction"].std()), 4),
                "stimulus_window_rate_mean": round(float(sub["stimulus_window_rate_hz"].mean()), 4),
                "post_stimulus_rate_mean": round(float(sub["post_stimulus_window_rate_hz"].mean()), 4),
                "activity_robustness_mean": round(float(act_rob_series.mean()), 4) if not act_rob_series.empty else np.nan,
                "activity_robustness_std": round(float(act_rob_series.std()), 4) if not act_rob_series.empty else np.nan,
                "edge_retention_mean": round(float(sub["edge_retention"].mean()), 4),
                "weight_retention_mean": round(float(sub["synaptic_weight_retention"].mean()), 4),
            })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = tables_dir / "stimulus_robustness_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Exported summary table to: {summary_csv}")

    # =========================================================================
    # STEP 5: GENERATE FIGURES
    # =========================================================================
    logger.info("\n--- STEP 5: Generating Publication Figures ---")
    generate_phase5_figures(summary_df, raw_df, figures_dir)

    # =========================================================================
    # STEP 6: PRINT EXPERIMENT SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 5: EXPERIMENT EXECUTION REPORT")
    print("=" * 80)
    print(f" Total Simulations:            {len(raw_df)} (5 Intact + 15 Hub + 75 Random)")
    print(f" Total Wall-Clock Runtime:     {total_runtime:.2f} seconds")
    print(f" Peak Memory Overhead:         {peak_ram_mb:.2f} MB")
    print("-" * 80)
    print(" INTACT BASELINE METRICS BY STIMULUS FRACTION:")
    for sf in stimulus_fractions:
        b = intact_baselines[sf]
        print(f"   Stim {sf*100:>2.0f}%: Rate = {b['mean_firing_rate']:>5.2f} Hz | Spikes = {b['total_spikes']:>5d} | Active = {b['active_neuron_fraction']*100:>4.1f}% | StimRate = {b['stimulus_window_rate_hz']:>5.2f} Hz | PostRate = {b['post_stimulus_window_rate_hz']:>5.2f} Hz")
    print("-" * 80)
    print(" ACTIVITY ROBUSTNESS AT 20% LESION:")
    print(f"{'Stim Frac':<12} {'Hub ActRob (20%)':<20} {'Random ActRob (20%)':<22} {'Hub Rate (Hz)':<16} {'Random Rate (Hz)':<18}")
    print("-" * 88)
    for sf in stimulus_fractions:
        h_row = summary_df[(summary_df["lesion_strategy"] == "hub") & (np.isclose(summary_df["stimulus_fraction"], sf)) & (np.isclose(summary_df["lesion_fraction"], 0.20))]
        r_row = summary_df[(summary_df["lesion_strategy"] == "random") & (np.isclose(summary_df["stimulus_fraction"], sf)) & (np.isclose(summary_df["lesion_fraction"], 0.20))]
        h_rob = f"{h_row['activity_robustness_mean'].iloc[0]:.3f}" if not h_row.empty and not np.isnan(h_row['activity_robustness_mean'].iloc[0]) else "NaN"
        r_rob = f"{r_row['activity_robustness_mean'].iloc[0]:.3f} ± {r_row['activity_robustness_std'].iloc[0]:.3f}" if not r_row.empty and not np.isnan(r_row['activity_robustness_mean'].iloc[0]) else "NaN"
        h_rate = f"{h_row['mean_firing_rate_mean'].iloc[0]:.2f}" if not h_row.empty else "N/A"
        r_rate = f"{r_row['mean_firing_rate_mean'].iloc[0]:.2f} ± {r_row['mean_firing_rate_std'].iloc[0]:.2f}" if not r_row.empty else "N/A"
        print(f"{sf*100:>2.0f}%{'':<9} {h_rob:<20} {r_rob:<22} {h_rate:<16} {r_rate:<18}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
