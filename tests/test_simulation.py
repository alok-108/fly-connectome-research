"""
Unit tests for LIF simulation module (src/simulation.py).
Verifies:
1. Simulation runs on a tiny 10-neuron network.
2. Membrane potentials remain finite (no NaNs, no Infs).
3. Spikes are valid discrete events.
4. Deterministic random seed produces exact reproducible results.
5. Zero external current behaves sensibly (subthreshold quiet state).
6. Neuron count and IDs are preserved in result structures.
7. Lesion-compatible network input works without error.
"""

import networkx as nx
import numpy as np
import pytest

from src.data_loader import generate_mock_connectome
from src.graph_builder import build_graph, get_subgraph
from src.lesion import apply_random_lesion
from src.simulation import LIFConfig, LIFNetwork, compute_population_rate, run_simulation


@pytest.fixture
def tiny_network():
    """Builds a deterministic 10-neuron directed test network."""
    G = nx.DiGraph()
    for i in range(10):
        G.add_node(i, cell_type="test_type", roi="test_roi")

    # Add ring connections with weight 5
    for i in range(10):
        G.add_edge(i, (i + 1) % 10, weight=5.0, roi="test_roi")
    return G


def test_tiny_network_simulation_runs(tiny_network):
    """Test 1: Simulation runs cleanly on a 10-neuron network."""
    config = LIFConfig(duration=100.0, dt=0.5, external_current=15.0, synaptic_weight_scale=0.01)
    res = run_simulation(tiny_network, config)

    assert res.is_stable is True
    assert len(res.neuron_ids) == 10
    assert len(res.spike_counts) == 10
    assert len(res.firing_rates) == 10
    assert res.spike_counts.sum() >= 0


def test_membrane_potentials_remain_finite(tiny_network):
    """Test 2: Membrane potentials and rates never contain NaN or Inf."""
    config = LIFConfig(duration=200.0, dt=0.5, external_current=20.0, synaptic_weight_scale=0.05, noise_sigma=1.5)
    network = LIFNetwork(tiny_network)
    res = network.simulate(config, record_samples=5)

    assert res.membrane_samples is not None
    assert not np.isnan(res.membrane_samples).any()
    assert not np.isinf(res.membrane_samples).any()
    assert (res.membrane_samples >= config.v_reset - 5.0).all()
    assert (res.membrane_samples <= 50.0).all()


def test_spikes_are_binary_events(tiny_network):
    """Test 3: Spike timestamps are non-negative, sorted, and map to valid neuron IDs."""
    config = LIFConfig(duration=150.0, dt=0.5, external_current=18.0)
    res = run_simulation(tiny_network, config)

    assert isinstance(res.spike_times, list)
    valid_ids = set(tiny_network.nodes())

    prev_t = -1.0
    for t_spike, nid in res.spike_times:
        assert 0.0 <= t_spike <= config.duration
        assert nid in valid_ids
        assert t_spike >= prev_t
        prev_t = t_spike


def test_deterministic_seed_reproducibility(tiny_network):
    """Test 4: Same seed yields identical spike counts, different seeds differ with noise."""
    cfg1 = LIFConfig(duration=150.0, dt=0.5, random_seed=101, noise_sigma=1.0)
    cfg2 = LIFConfig(duration=150.0, dt=0.5, random_seed=101, noise_sigma=1.0)
    cfg3 = LIFConfig(duration=150.0, dt=0.5, random_seed=202, noise_sigma=1.0)

    res1 = run_simulation(tiny_network, cfg1)
    res2 = run_simulation(tiny_network, cfg2)
    res3 = run_simulation(tiny_network, cfg3)

    np.testing.assert_array_equal(res1.spike_counts, res2.spike_counts)
    assert res1.spike_times == res2.spike_times


def test_zero_external_current_subthreshold(tiny_network):
    """Test 5: With zero external current and zero noise, network remains quiet at rest."""
    config = LIFConfig(duration=100.0, dt=0.5, external_current=0.0, noise_sigma=0.0, synaptic_weight_scale=0.01)
    res = run_simulation(tiny_network, config)

    assert res.spike_counts.sum() == 0
    assert len(res.spike_times) == 0
    assert (res.firing_rates == 0.0).all()


def test_neuron_count_preservation(tiny_network):
    """Test 6: Neuron count is strictly preserved across inputs and outputs."""
    config = LIFConfig(duration=50.0, dt=0.5)
    res = run_simulation(tiny_network, config)

    assert len(res.neuron_ids) == tiny_network.number_of_nodes()
    assert len(res.firing_rates) == tiny_network.number_of_nodes()


def test_lesion_compatible_network_input():
    """Test 7: Lesioned graphs (with removed nodes) simulate cleanly without key errors."""
    nodes_df, edges_df = generate_mock_connectome(num_neurons=50, avg_degree=6.0, random_seed=42)
    G = build_graph(nodes_df, edges_df)

    # Apply 20% random lesion
    G_lesioned, removed_nodes = apply_random_lesion(G, lesion_fraction=0.20, seed=42)
    assert G_lesioned.number_of_nodes() == 40

    config = LIFConfig(duration=100.0, dt=0.5, external_current=14.0)
    res = run_simulation(G_lesioned, config)

    assert res.is_stable is True
    assert len(res.neuron_ids) == 40
    # Ensure none of the removed nodes appear in results
    assert set(removed_nodes).intersection(set(res.neuron_ids)) == set()
