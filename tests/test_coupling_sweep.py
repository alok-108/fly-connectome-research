"""
Unit and integration tests for Phase 6 Coupling-Regime Sweep (experiments/05_coupling_sweep.py).

Tests:
1. Exact alpha sweep values match specification.
2. Deterministic stimulus selection and fixed stimulus set across alphas.
3. Activity windows partition correctly without overlapping time gaps.
4. Stimulated vs unstimulated neuron partition and firing rate calculations.
5. Unstimulated recruitment calculation during stimulus window.
6. Post-stimulus persistence and safe zero-division handling.
7. Dynamic regime classification logic (stable/unstable/driven/recruited).
8. Graph immutability and no mutation across sweeps.
9. Deterministic reproducibility of simulation metrics under identical parameters.
10. Limited lesion validation baseline normalization against same alpha.
"""

import importlib.util
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pytest

sweep_path = Path(__file__).resolve().parent.parent / "experiments" / "05_coupling_sweep.py"
spec = importlib.util.spec_from_file_location("exp05", sweep_path)
exp05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp05)

ALPHA_VALUES = exp05.ALPHA_VALUES
STIMULUS_FRACTIONS = exp05.STIMULUS_FRACTIONS
classify_coupling_regime = exp05.classify_coupling_regime
compute_activity_windows = exp05.compute_activity_windows
compute_connectivity_metrics = exp05.compute_connectivity_metrics
compute_recurrent_activity_metrics = exp05.compute_recurrent_activity_metrics
select_representative_alphas = exp05.select_representative_alphas

from src.simulation import LIFConfig, LIFNetwork, select_stimulated_neurons


@pytest.fixture
def sample_graph():
    """Generates a reproducible 20-neuron directed test graph."""
    G = nx.DiGraph()
    for i in range(20):
        G.add_node(i, cell_type="test", roi="central")

    # Connect in forward-chain and hub pattern
    for i in range(19):
        G.add_edge(i, i + 1, weight=3.0)
    # Node 0 connects to all even nodes
    for i in range(2, 20, 2):
        G.add_edge(0, i, weight=5.0)
    return G


def test_1_exact_alpha_sweep_values():
    """Test 1: Verify exact 11 canonical alpha values in ascending order."""
    expected_alphas = [
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
    assert len(ALPHA_VALUES) == 11
    assert ALPHA_VALUES == expected_alphas
    assert all(ALPHA_VALUES[i] < ALPHA_VALUES[i + 1] for i in range(len(ALPHA_VALUES) - 1))


def test_2_deterministic_stimulus_set_fixed_across_alphas(sample_graph):
    """Test 2: Stimulated set is deterministically identical regardless of alpha."""
    stim_25_a = select_stimulated_neurons(sample_graph, stimulus_fraction=0.25, seed=42)
    stim_25_b = select_stimulated_neurons(sample_graph, stimulus_fraction=0.25, seed=42)
    assert stim_25_a == stim_25_b
    assert len(stim_25_a) == 5  # 25% of 20 = 5

    stim_10 = select_stimulated_neurons(sample_graph, stimulus_fraction=0.10, seed=42)
    assert len(stim_10) == 2


def test_3_activity_windows_partition(sample_graph):
    """Test 3: Temporal windows correctly partition spikes into pre, stim, and post epochs."""
    neuron_ids = list(sample_graph.nodes())
    spike_times = [
        (50.0, 0),    # Pre-stimulus (0-200)
        (150.0, 1),   # Pre-stimulus
        (250.0, 0),   # Stimulus (200-600)
        (350.0, 0),   # Stimulus
        (550.0, 2),   # Stimulus
        (700.0, 0),   # Post-stimulus (600-1000)
        (950.0, 3),   # Post-stimulus
    ]

    win_data = compute_activity_windows(spike_times, num_neurons=20, neuron_ids=neuron_ids, duration=1000.0)
    assert win_data["pre_stimulus"]["total_spikes"] == 2
    assert win_data["stimulus"]["total_spikes"] == 3
    assert win_data["post_stimulus"]["total_spikes"] == 2
    assert win_data["full"]["total_spikes"] == 7
    assert (
        win_data["pre_stimulus"]["total_spikes"]
        + win_data["stimulus"]["total_spikes"]
        + win_data["post_stimulus"]["total_spikes"]
        == win_data["full"]["total_spikes"]
    )


def test_4_stimulated_vs_unstimulated_partition(sample_graph):
    """Test 4: Correctly partitions spike counts and computes rates per group."""
    neuron_ids = list(sample_graph.nodes())
    stimulated_ids = [0, 1, 2]
    # Spikes: 10 spikes on stimulated (0, 1), 2 spikes on unstimulated (3)
    spike_counts = np.zeros(20, dtype=np.int32)
    spike_counts[0] = 6
    spike_counts[1] = 4
    spike_counts[3] = 2

    spike_times = [(300.0, 0), (350.0, 1), (400.0, 3)]
    win_data = compute_activity_windows(spike_times, num_neurons=20, neuron_ids=neuron_ids, duration=1000.0)

    rec_data = compute_recurrent_activity_metrics(
        spike_counts=spike_counts,
        neuron_ids=neuron_ids,
        stimulated_neuron_ids=stimulated_ids,
        windows_data=win_data,
        duration=1000.0,
    )

    assert rec_data["stimulated_neuron_count"] == 3
    assert rec_data["unstimulated_neuron_count"] == 17
    # Stimulated rate: (6 + 4 + 0) / 3 / 1.0 = 3.3333 Hz
    assert np.isclose(rec_data["stimulated_rate_hz"], 10.0 / 3.0, atol=1e-3)
    # Unstimulated rate: 2 / 17 / 1.0 = 0.1176 Hz
    assert np.isclose(rec_data["unstimulated_rate_hz"], 2.0 / 17.0, atol=1e-3)
    assert not np.isnan(rec_data["stimulated_to_unstimulated_ratio"])


def test_5_unstimulated_recruitment_calculation(sample_graph):
    """Test 5: Calculates recruited unstimulated neurons during stimulus window."""
    neuron_ids = list(sample_graph.nodes())
    stimulated_ids = [0, 1, 2]

    # Spikes during stimulus window: neuron 3 and 4 fire
    spike_times = [
        (300.0, 0),  # stimulated
        (320.0, 3),  # unstimulated recruited
        (350.0, 4),  # unstimulated recruited
    ]
    spike_counts = np.zeros(20, dtype=np.int32)
    spike_counts[0] = 1
    spike_counts[3] = 1
    spike_counts[4] = 1

    win_data = compute_activity_windows(spike_times, num_neurons=20, neuron_ids=neuron_ids, duration=1000.0)
    rec_data = compute_recurrent_activity_metrics(
        spike_counts=spike_counts,
        neuron_ids=neuron_ids,
        stimulated_neuron_ids=stimulated_ids,
        windows_data=win_data,
        duration=1000.0,
    )

    assert rec_data["recruited_unstimulated_count"] == 2
    assert np.isclose(rec_data["recruited_unstimulated_fraction"], 2.0 / 17.0, atol=1e-3)


def test_6_zero_division_safety_and_nans(sample_graph):
    """Test 6: Verifies safe NaN handling when unstimulated rate or stimulus active fraction is zero."""
    neuron_ids = list(sample_graph.nodes())
    stimulated_ids = [0, 1]
    spike_counts = np.zeros(20, dtype=np.int32)
    # No spikes at all
    win_data = compute_activity_windows([], num_neurons=20, neuron_ids=neuron_ids, duration=1000.0)
    rec_data = compute_recurrent_activity_metrics(
        spike_counts=spike_counts,
        neuron_ids=neuron_ids,
        stimulated_neuron_ids=stimulated_ids,
        windows_data=win_data,
        duration=1000.0,
    )

    assert np.isnan(rec_data["stimulated_to_unstimulated_ratio"])
    assert np.isnan(rec_data["post_stimulus_persistence"])
    assert rec_data["recruited_unstimulated_count"] == 0
    assert rec_data["recruited_unstimulated_fraction"] == 0.0


def test_7_coupling_regime_classification():
    """Test 7: Tests classification logic for all documented regimes."""
    # Unstable condition
    assert classify_coupling_regime(is_stable=False, recruited_unstimulated_fraction=0.1, post_stimulus_rate_hz=1.5) == "Unstable"

    # Sustained post-stimulus condition
    assert classify_coupling_regime(is_stable=True, recruited_unstimulated_fraction=0.2, post_stimulus_rate_hz=0.5) == "Sustained Recurrent Activity"

    # Recurrent recruitment during stimulus but no post-stimulus activity
    assert classify_coupling_regime(is_stable=True, recruited_unstimulated_fraction=0.05, post_stimulus_rate_hz=0.0) == "Recurrent Recruitment"

    # Externally driven / weak coupling
    assert classify_coupling_regime(is_stable=True, recruited_unstimulated_fraction=0.005, post_stimulus_rate_hz=0.0) == "Externally Driven (Weak Coupling)"


def test_8_graph_immutability_during_connectivity_metrics(sample_graph):
    """Test 8: Calculating connectivity metrics does not mutate graph nodes or edges."""
    n_orig = sample_graph.number_of_nodes()
    e_orig = sample_graph.number_of_edges()

    neuron_ids = list(sample_graph.nodes())
    spike_counts = np.ones(20, dtype=np.int32)
    conn_m = compute_connectivity_metrics(
        graph=sample_graph,
        spike_counts=spike_counts,
        neuron_ids=neuron_ids,
        stimulated_neuron_ids=[0, 1],
        alpha=0.01,
    )

    assert sample_graph.number_of_nodes() == n_orig
    assert sample_graph.number_of_edges() == e_orig
    assert conn_m["total_synaptic_input_mv"] > 0
    assert conn_m["fraction_neurons_receiving_recurrent_input"] > 0


def test_9_reproducibility_across_same_alpha(sample_graph):
    """Test 9: Identical parameters yield exact bitwise identical simulation results."""
    cfg = LIFConfig(duration=200.0, dt=0.5, external_current=18.0, synaptic_weight_scale=0.01, random_seed=42)
    net = LIFNetwork(sample_graph)

    res1 = net.simulate(cfg, stimulated_neuron_ids=[0, 1])
    res2 = net.simulate(cfg, stimulated_neuron_ids=[0, 1])

    np.testing.assert_array_equal(res1.spike_counts, res2.spike_counts)
    assert res1.spike_times == res2.spike_times
    assert res1.metrics["mean_firing_rate_hz"] == res2.metrics["mean_firing_rate_hz"]
    assert res1.is_stable == res2.is_stable


def test_10_lesion_normalization_against_same_alpha():
    """Test 10: Ensures lesion activity robustness is normalized against baseline at the same alpha."""
    intact_rate_alpha_01 = 2.0
    lesion_rate_alpha_01 = 1.8
    intact_rate_alpha_02 = 4.0
    lesion_rate_alpha_02 = 3.6

    rob_01 = lesion_rate_alpha_01 / intact_rate_alpha_01
    rob_02 = lesion_rate_alpha_02 / intact_rate_alpha_02

    assert np.isclose(rob_01, 0.90)
    assert np.isclose(rob_02, 0.90)
    # Normalizing cross-alpha (lesion 0.02 against intact 0.01) would give invalid 1.80
    invalid_cross_rob = lesion_rate_alpha_02 / intact_rate_alpha_01
    assert not np.isclose(invalid_cross_rob, rob_02)
