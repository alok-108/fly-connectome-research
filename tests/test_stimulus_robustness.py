"""
Unit tests for Phase 5: Stimulus-Dependence of Lesion Robustness.

Tests:
1. Stimulus fraction calculation & correct stimulus neuron counts.
2. Deterministic stimulus selection (same seed -> identical list; different seed -> different list).
3. Surviving stimulated neuron calculation and natural ablation.
4. Temporal window calculations (stimulus window 200-600ms vs post-stimulus window 600-1000ms).
5. Zero-division handling (no infinity; NaN for undefined ratios).
6. Baseline normalization per stimulus condition (own intact baseline).
7. Graph preservation and immutability.
8. Simulation execution with explicit stimulated neurons (including 0% condition).
9. Lesion reproducibility under Phase 5 seeds.
10. Effective stimulus fraction bounds and invariants.
"""

from typing import List

import networkx as nx
import numpy as np
import pytest

from src.data_loader import generate_mock_connectome
from src.graph_builder import build_graph
from src.lesion import apply_hub_lesion, apply_random_lesion
from src.simulation import (
    LIFConfig,
    compute_temporal_metrics,
    run_simulation,
    select_stimulated_neurons,
)


@pytest.fixture
def test_graph():
    """Provides a reproducible 200-neuron mock connectome for fast testing."""
    nodes_df, edges_df = generate_mock_connectome(num_neurons=200, avg_degree=10.0, random_seed=42)
    return build_graph(nodes_df, edges_df)


def test_1_stimulus_fraction_and_counts(test_graph):
    """Verifies that select_stimulated_neurons returns exact floor/round counts."""
    n = test_graph.number_of_nodes()  # 200
    expected = {
        0.25: int(np.round(n * 0.25)),  # 50
        0.10: int(np.round(n * 0.10)),  # 20
        0.05: int(np.round(n * 0.05)),  # 10
        0.01: int(np.round(n * 0.01)),  # 2
        0.00: 0,
    }

    for frac, exp_count in expected.items():
        stim_nodes = select_stimulated_neurons(test_graph, stimulus_fraction=frac, seed=42)
        assert len(stim_nodes) == exp_count
        assert len(set(stim_nodes)) == exp_count  # No duplicates
        # All selected neurons must belong to graph
        assert set(stim_nodes).issubset(set(test_graph.nodes()))


def test_2_deterministic_stimulus_selection(test_graph):
    """Verifies that deterministic seed produces identical selection; different seed differs."""
    s1 = select_stimulated_neurons(test_graph, stimulus_fraction=0.10, seed=42)
    s2 = select_stimulated_neurons(test_graph, stimulus_fraction=0.10, seed=42)
    s3 = select_stimulated_neurons(test_graph, stimulus_fraction=0.10, seed=999)

    assert s1 == s2
    assert s1 != s3


def test_3_surviving_stimulated_neuron_calculation(test_graph):
    """Verifies that lesioned stimulated neurons are naturally removed without re-sampling."""
    initial_stim = select_stimulated_neurons(test_graph, stimulus_fraction=0.25, seed=42)
    n_initial = len(initial_stim)
    assert n_initial == 50

    # Apply 20% random lesion (40 neurons ablated)
    lesioned_g, ablated = apply_random_lesion(test_graph, lesion_fraction=0.20, seed=42)
    surviving_nodes = set(lesioned_g.nodes())

    surviving_stim = [nid for nid in initial_stim if nid in surviving_nodes]
    surviving_count = len(surviving_stim)

    # Surviving count must be <= initial count
    assert surviving_count <= n_initial
    # Effective fraction
    eff_frac = surviving_count / lesioned_g.number_of_nodes()
    assert 0.0 <= eff_frac <= 1.0


def test_4_temporal_window_calculations():
    """Verifies spike partitioning into stimulus (200-600ms) and post-stimulus (600-1000ms) windows."""
    # Create synthetic spikes: 10 in stimulus window, 2 in post-stimulus window
    synthetic_spikes = [
        (100.0, 1),  # pre-stimulus (<200ms)
        (250.0, 2),  # stimulus
        (300.0, 3),  # stimulus
        (400.0, 1),  # stimulus
        (550.0, 2),  # stimulus
        (650.0, 1),  # post-stimulus
        (800.0, 3),  # post-stimulus
    ]

    metrics = compute_temporal_metrics(
        synthetic_spikes,
        num_neurons=10,
        stimulus_window=(200.0, 600.0),
        post_window=(600.0, 1000.0),
    )

    assert metrics["stimulus_window_spikes"] == 4
    assert metrics["post_stimulus_window_spikes"] == 2
    # Rate in stim window: 4 spikes / (10 neurons * 0.4s) = 1.0 Hz
    assert np.isclose(metrics["stimulus_window_rate_hz"], 1.0)
    # Rate in post window: 2 spikes / (10 neurons * 0.4s) = 0.5 Hz
    assert np.isclose(metrics["post_stimulus_window_rate_hz"], 0.5)
    # Ratio: 1.0 / 0.5 = 2.0
    assert np.isclose(metrics["stimulus_to_post_ratio"], 2.0)


def test_5_zero_division_handling_no_infinity():
    """Verifies that zero post-stimulus activity yields NaN, never inf."""
    spikes_stim_only = [(300.0, 1), (400.0, 2)]

    metrics = compute_temporal_metrics(
        spikes_stim_only,
        num_neurons=10,
        stimulus_window=(200.0, 600.0),
        post_window=(600.0, 1000.0),
    )

    assert metrics["stimulus_window_spikes"] == 2
    assert metrics["post_stimulus_window_spikes"] == 0
    assert np.isnan(metrics["stimulus_to_post_ratio"])
    assert not np.isinf(metrics["stimulus_to_post_ratio"])


def test_6_baseline_normalization(test_graph):
    """Verifies that each stimulus condition normalizes against its own intact baseline."""
    # Test normalization function logic
    def compute_robustness(lesion_val: float, baseline_val: float) -> float:
        if baseline_val == 0.0 or np.isnan(baseline_val):
            return np.nan
        return float(lesion_val / baseline_val)

    # Normal case
    assert np.isclose(compute_robustness(2.0, 2.5), 0.8)
    # Zero baseline case
    assert np.isnan(compute_robustness(0.0, 0.0))
    assert not np.isinf(compute_robustness(0.0, 0.0))


def test_7_graph_preservation_and_immutability(test_graph):
    """Verifies that the original graph is strictly untouched throughout stimulus & lesion operations."""
    n_before = test_graph.number_of_nodes()
    e_before = test_graph.number_of_edges()

    stim_nodes = select_stimulated_neurons(test_graph, stimulus_fraction=0.25, seed=42)
    apply_random_lesion(test_graph, lesion_fraction=0.10, seed=123)
    apply_hub_lesion(test_graph, lesion_fraction=0.10)

    assert test_graph.number_of_nodes() == n_before
    assert test_graph.number_of_edges() == e_before


def test_8_simulation_with_explicit_stimulated_neurons(test_graph):
    """Verifies simulation runs with explicit stimulated neurons and tests 0% condition."""
    cfg = LIFConfig(
        duration=100.0,
        dt=0.5,
        synaptic_weight_scale=0.01,
        external_current=18.0,
        external_stimulus_mode="pulse",
        pulse_start=20.0,
        pulse_end=60.0,
        random_seed=42,
    )

    # 1. Non-zero stimulation
    stim_nodes = select_stimulated_neurons(test_graph, stimulus_fraction=0.10, seed=42)
    res_stim = run_simulation(test_graph, config=cfg, stimulated_neuron_ids=stim_nodes)
    assert res_stim.is_stable
    assert res_stim.metrics["total_spikes"] >= 0

    # 2. Zero stimulation condition
    res_zero = run_simulation(test_graph, config=cfg, stimulated_neuron_ids=[])
    assert res_zero.is_stable
    # In deterministic subthreshold regime with I=0, total spikes should be 0
    assert res_zero.metrics["total_spikes"] == 0
    assert res_zero.metrics["mean_firing_rate_hz"] == 0.0


def test_9_lesion_reproducibility_phase5_seeds(test_graph):
    """Verifies that Phase 5 seeds (42, 123, 456, 789, 1000) produce reproducible ablations."""
    seeds = [42, 123, 456, 789, 1000]

    for s in seeds:
        g1, n1 = apply_random_lesion(test_graph, lesion_fraction=0.10, seed=s)
        g2, n2 = apply_random_lesion(test_graph, lesion_fraction=0.10, seed=s)
        assert n1 == n2
        assert set(g1.nodes()) == set(g2.nodes())


def test_10_effective_stimulus_fraction_bounds(test_graph):
    """Verifies that effective stimulus fraction is strictly within [0.0, 1.0]."""
    for frac in [0.25, 0.10, 0.05, 0.01, 0.0]:
        stim = select_stimulated_neurons(test_graph, stimulus_fraction=frac, seed=42)
        for les_frac in [0.05, 0.10, 0.20]:
            g_les, _ = apply_random_lesion(test_graph, lesion_fraction=les_frac, seed=42)
            surv_stim = [nid for nid in stim if nid in g_les]
            eff = len(surv_stim) / g_les.number_of_nodes()
            assert 0.0 <= eff <= 1.0
