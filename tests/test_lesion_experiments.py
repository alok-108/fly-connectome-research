"""
Unit and integration tests for virtual lesion experiments (Phase 4).

Verifies the 10 core scientific invariants:
1. Correct lesion counts (exactly floor(fraction * N)).
2. Random lesion reproducibility.
3. Hub lesion removes highest-degree neurons.
4. Original graph is not mutated.
5. Lesion never increases neuron count.
6. Lesion never increases edge count.
7. Simulation accepts all lesion graphs.
8. Baseline graph remains unchanged across full workflow.
9. Same seed produces same random lesion.
10. Different random seeds produce different lesion sets.
"""

import networkx as nx
import numpy as np
import pytest

from src.data_loader import generate_mock_connectome
from src.graph_builder import build_graph, calculate_degree
from src.lesion import (
    apply_hub_lesion,
    apply_random_lesion,
    compute_lesion_graph_metrics,
)
from src.simulation import LIFConfig, run_simulation


@pytest.fixture
def sample_connectome():
    """Provides a reproducible 150-neuron directed test connectome with heterogeneous degrees."""
    nodes_df, edges_df = generate_mock_connectome(num_neurons=150, avg_degree=12.0, random_seed=42)
    return build_graph(nodes_df, edges_df)


def test_1_correct_lesion_counts(sample_connectome):
    """Test 1: Both random and hub lesions remove exactly floor(fraction * N) neurons."""
    n = sample_connectome.number_of_nodes()

    for frac in [0.01, 0.05, 0.10, 0.20]:
        expected_remove = int(np.floor(frac * n))

        # Random lesion
        _, rnd_nodes = apply_random_lesion(sample_connectome, lesion_fraction=frac, seed=42)
        assert len(rnd_nodes) == expected_remove

        # Hub lesion
        _, hub_nodes = apply_hub_lesion(sample_connectome, lesion_fraction=frac)
        assert len(hub_nodes) == expected_remove


def test_2_random_lesion_reproducibility(sample_connectome):
    """Test 2: Random lesion with fixed seed produces identical output repeatedly."""
    g1, nodes1 = apply_random_lesion(sample_connectome, lesion_fraction=0.10, seed=123)
    g2, nodes2 = apply_random_lesion(sample_connectome, lesion_fraction=0.10, seed=123)

    assert nodes1 == nodes2
    assert list(g1.nodes()) == list(g2.nodes())
    assert list(g1.edges()) == list(g2.edges())


def test_3_hub_lesion_removes_highest_degree(sample_connectome):
    """Test 3: Hub lesion removes the highest pre-lesion degree neurons with deterministic tie-breaking."""
    degrees = calculate_degree(sample_connectome, weighted=False)
    # Sort descending by degree, ascending by node_id for ties
    sorted_nodes = sorted(degrees.items(), key=lambda x: (-x[1], x[0]))

    for frac in [0.05, 0.10]:
        k = int(np.floor(frac * sample_connectome.number_of_nodes()))
        expected_hubs = [node for node, _ in sorted_nodes[:k]]

        _, ablated_hubs = apply_hub_lesion(sample_connectome, lesion_fraction=frac, weighted=False)
        assert ablated_hubs == expected_hubs


def test_4_original_graph_is_not_mutated(sample_connectome):
    """Test 4: Original graph nodes, edges, and attributes remain strictly invariant after lesioning."""
    nodes_before = list(sample_connectome.nodes(data=True))
    edges_before = list(sample_connectome.edges(data=True))

    # Apply multiple lesions
    apply_random_lesion(sample_connectome, lesion_fraction=0.20, seed=42)
    apply_hub_lesion(sample_connectome, lesion_fraction=0.20)

    nodes_after = list(sample_connectome.nodes(data=True))
    edges_after = list(sample_connectome.edges(data=True))

    assert nodes_before == nodes_after
    assert edges_before == edges_after


def test_5_lesion_never_increases_neuron_count(sample_connectome):
    """Test 5: Lesion never increases neuron count."""
    n_orig = sample_connectome.number_of_nodes()

    for frac in [0.0, 0.01, 0.05, 0.10, 0.20]:
        g_rnd, _ = apply_random_lesion(sample_connectome, lesion_fraction=frac, seed=42)
        assert g_rnd.number_of_nodes() <= n_orig

        g_hub, _ = apply_hub_lesion(sample_connectome, lesion_fraction=frac)
        assert g_hub.number_of_nodes() <= n_orig


def test_6_lesion_never_increases_edge_count(sample_connectome):
    """Test 6: Lesion never increases edge count and all remaining edges are from original set."""
    e_orig = sample_connectome.number_of_edges()
    orig_edge_set = set(sample_connectome.edges())

    for frac in [0.01, 0.05, 0.10, 0.20]:
        g_rnd, _ = apply_random_lesion(sample_connectome, lesion_fraction=frac, seed=42)
        assert g_rnd.number_of_edges() <= e_orig
        assert set(g_rnd.edges()).issubset(orig_edge_set)

        g_hub, _ = apply_hub_lesion(sample_connectome, lesion_fraction=frac)
        assert g_hub.number_of_edges() <= e_orig
        assert set(g_hub.edges()).issubset(orig_edge_set)


def test_7_simulation_accepts_all_lesion_graphs(sample_connectome):
    """Test 7: LIF simulation runs stably on both random and hub lesioned graphs across all fractions."""
    cfg = LIFConfig(
        duration=100.0,
        dt=0.5,
        synaptic_weight_scale=0.01,
        external_current=15.0,
        external_stimulus_mode="pulse",
        pulse_start=20.0,
        pulse_end=80.0,
        random_seed=42,
    )

    for frac in [0.01, 0.05, 0.10, 0.20]:
        # Random lesion
        g_rnd, _ = apply_random_lesion(sample_connectome, lesion_fraction=frac, seed=42)
        res_rnd = run_simulation(g_rnd, config=cfg)
        assert res_rnd.is_stable
        assert len(res_rnd.neuron_ids) == g_rnd.number_of_nodes()

        # Hub lesion
        g_hub, _ = apply_hub_lesion(sample_connectome, lesion_fraction=frac)
        res_hub = run_simulation(g_hub, config=cfg)
        assert res_hub.is_stable
        assert len(res_hub.neuron_ids) == g_hub.number_of_nodes()


def test_8_baseline_graph_remains_unchanged(sample_connectome):
    """Test 8: Baseline graph metrics remain unchanged after calculating comparative lesion metrics."""
    metrics_before = compute_lesion_graph_metrics(
        sample_connectome, sample_connectome, lesion_strategy="baseline", lesion_fraction=0.0
    )

    # Perform lesions and metrics
    g_rnd, _ = apply_random_lesion(sample_connectome, lesion_fraction=0.10, seed=42)
    compute_lesion_graph_metrics(sample_connectome, g_rnd, lesion_strategy="random", lesion_fraction=0.10, seed=42)

    g_hub, _ = apply_hub_lesion(sample_connectome, lesion_fraction=0.10)
    compute_lesion_graph_metrics(sample_connectome, g_hub, lesion_strategy="hub", lesion_fraction=0.10)

    metrics_after = compute_lesion_graph_metrics(
        sample_connectome, sample_connectome, lesion_strategy="baseline", lesion_fraction=0.0
    )

    assert metrics_before == metrics_after


def test_9_same_seed_produces_same_random_lesion(sample_connectome):
    """Test 9: Calling apply_random_lesion with the same seed always ablates the exact same neuron IDs."""
    for seed in [42, 123, 456, 789, 1000]:
        _, nodes_run1 = apply_random_lesion(sample_connectome, lesion_fraction=0.10, seed=seed)
        _, nodes_run2 = apply_random_lesion(sample_connectome, lesion_fraction=0.10, seed=seed)
        assert nodes_run1 == nodes_run2


def test_10_different_random_seeds_produce_different_lesion_sets(sample_connectome):
    """Test 10: Different seeds produce distinct sets of ablated neurons."""
    seeds = [42, 123, 456, 789, 1000]
    lesion_sets = []

    for seed in seeds:
        _, nodes = apply_random_lesion(sample_connectome, lesion_fraction=0.10, seed=seed)
        lesion_sets.append(set(nodes))

    # Verify that not all sets are identical (at least pairwise differences exist)
    unique_sets = set(tuple(sorted(s)) for s in lesion_sets)
    assert len(unique_sets) == len(seeds), "Each seed must produce a unique lesion set!"
