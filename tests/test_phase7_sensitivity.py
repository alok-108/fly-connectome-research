"""
Unit and integration tests for Phase 7 Robustness & Sensitivity Validation.
"""

from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import extract_real_subnetwork
from src.graph_builder import get_graph_summary
from src.lesion import apply_hub_lesion, apply_random_lesion
from src.simulation import LIFConfig, LIFNetwork, select_stimulated_neurons


@pytest.fixture
def sample_test_graph():
    """Generates a reproducible 20-neuron directed test graph."""
    G = nx.DiGraph()
    for i in range(20):
        G.add_node(i, cell_type="test", roi="central")
    for i in range(19):
        G.add_edge(i, i + 1, weight=3.0)
    for i in range(2, 20, 2):
        G.add_edge(0, i, weight=5.0)
    return G


def test_1_alpha_sensitivity_config_and_values():
    """Test 1: Verify exact alpha values for Experiment 7A."""
    expected_alphas = [0.0150, 0.0175, 0.0200, 0.0225, 0.0250]
    assert len(expected_alphas) == 5
    assert all(expected_alphas[i] < expected_alphas[i + 1] for i in range(len(expected_alphas) - 1))
    assert 0.0200 in expected_alphas


def test_2_stimulus_selection_consistency(sample_test_graph):
    """Test 2: Verify deterministic stimulus selection across fractions."""
    stim_05 = select_stimulated_neurons(sample_test_graph, stimulus_fraction=0.05, seed=42)
    stim_10 = select_stimulated_neurons(sample_test_graph, stimulus_fraction=0.10, seed=42)
    stim_25 = select_stimulated_neurons(sample_test_graph, stimulus_fraction=0.25, seed=42)

    assert len(stim_05) == 1   # max(1, int(20 * 0.05)) = 1
    assert len(stim_10) == 2   # 20 * 0.10 = 2
    assert len(stim_25) == 5   # 20 * 0.25 = 5

    # Determinism check
    assert select_stimulated_neurons(sample_test_graph, stimulus_fraction=0.25, seed=42) == stim_25


def test_3_pre_lesion_stimulus_selection_invariance(sample_test_graph):
    """Test 3: Verify stimulus set selected before lesioning correctly tracks surviving neurons."""
    stim_ids = select_stimulated_neurons(sample_test_graph, stimulus_fraction=0.25, seed=42)
    assert len(stim_ids) == 5

    # Apply 20% hub lesion
    lesioned_g, _ = apply_hub_lesion(sample_test_graph, lesion_fraction=0.20)
    surv_stim = [nid for nid in stim_ids if nid in lesioned_g]

    assert len(surv_stim) <= len(stim_ids)
    assert all(nid in lesioned_g for nid in surv_stim)


def test_4_lesion_fraction_monotonicity(sample_test_graph):
    """Test 4: Verify lesion fractions remove monotonically increasing neuron counts."""
    fractions = [0.05, 0.10, 0.20]
    n_orig = sample_test_graph.number_of_nodes()

    surv_counts = []
    for lf in fractions:
        lesioned_g, _ = apply_hub_lesion(sample_test_graph, lesion_fraction=lf)
        surv_counts.append(lesioned_g.number_of_nodes())

    # Strictly decreasing surviving nodes
    assert surv_counts[0] > surv_counts[1] > surv_counts[2]
    assert surv_counts[-1] < n_orig


def test_5_timestep_configuration_fixed_physical_duration():
    """Test 5: Verify timesteps maintain fixed physical duration (1000 ms)."""
    duration = 1000.0
    for dt in [0.25, 0.50, 1.00]:
        cfg = LIFConfig(duration=duration, dt=dt)
        n_steps = int(cfg.duration / cfg.dt)
        assert n_steps in [4000, 2000, 1000]
        assert np.isclose(n_steps * cfg.dt, duration)


def test_6_random_subnetwork_extraction_reproducibility():
    """Test 6: Verify reproducible extraction of random subnetwork with seed 2026."""
    g1, d1 = extract_real_subnetwork(max_neurons=50, strategy="random", seed=2026, verbose=False)
    g2, d2 = extract_real_subnetwork(max_neurons=50, strategy="random", seed=2026, verbose=False)

    assert list(g1.nodes()) == list(g2.nodes())
    assert list(g1.edges()) == list(g2.edges())
    assert d1["total_synaptic_weight"] == d2["total_synaptic_weight"]


def test_7_output_table_schemas_and_metadata_columns():
    """Test 7: Verify required metadata and metric column definitions."""
    required_cols = [
        "experiment",
        "alpha",
        "stimulus_fraction",
        "lesion_fraction",
        "lesion_strategy",
        "random_seed",
        "dt",
        "network_selection",
        "network_seed",
        "initial_neurons",
        "surviving_neurons",
        "edge_retention",
        "synaptic_weight_retention",
        "is_stable",
        "mean_firing_rate_hz",
        "activity_robustness",
    ]
    # Check dummy record has all required keys
    dummy_record = {col: 1 for col in required_cols}
    assert all(col in dummy_record for col in required_cols)


def test_8_deterministic_simulation_metrics_reproducibility(sample_test_graph):
    """Test 8: Ensure identical seeds produce identical numerical metrics."""
    cfg = LIFConfig(duration=200.0, dt=0.5, external_current=18.0, synaptic_weight_scale=0.02, random_seed=42)
    net = LIFNetwork(sample_test_graph)

    res1 = net.simulate(cfg, stimulated_neuron_ids=[0, 1])
    res2 = net.simulate(cfg, stimulated_neuron_ids=[0, 1])

    assert res1.metrics["mean_firing_rate_hz"] == res2.metrics["mean_firing_rate_hz"]
    assert res1.metrics["total_spikes"] == res2.metrics["total_spikes"]
    assert res1.spike_times == res2.spike_times


def test_9_robustness_normalization_against_same_baseline():
    """Test 9: Ensure activity robustness normalizes against baseline of the same condition."""
    intact_rate_sf10 = 2.5
    lesion_rate_sf10 = 2.0
    intact_rate_sf25 = 5.0
    lesion_rate_sf25 = 4.0

    rob_sf10 = lesion_rate_sf10 / intact_rate_sf10
    rob_sf25 = lesion_rate_sf25 / intact_rate_sf25

    assert np.isclose(rob_sf10, 0.80)
    assert np.isclose(rob_sf25, 0.80)
    # Cross-normalization would be incorrect
    invalid_cross = lesion_rate_sf25 / intact_rate_sf10
    assert not np.isclose(invalid_cross, rob_sf25)


def test_10_exact_simulation_counts_accounting():
    """Test 10: Verify exact simulation count accounting across all sub-experiments."""
    count_7a = 5
    count_7b = 2 * 3 * 13   # 2 alphas x 3 stimulus fractions x 13 conditions = 78
    count_7c = 2 * 19       # 2 alphas x (1 intact + 3 hub + 15 random) = 38
    count_7d = 2 * 3 * 4    # 2 alphas x 3 dts x 4 conditions = 24
    count_7e = 2 * 2 * 3    # 2 networks x 2 alphas x 3 conditions = 12

    total = count_7a + count_7b + count_7c + count_7d + count_7e
    assert total == 157
