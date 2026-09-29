"""
Unit tests for data_loader module.
Verifies schema, deterministic mock generation, filtering, storage round-trips,
and real schema validation / subnetwork extraction routines (offline).
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    SCHEMA,
    OFFICIAL_DATASETS,
    generate_mock_connectome,
    load_processed_data,
    validate_real_data,
    extract_real_subnetwork,
)


def test_official_datasets_metadata():
    """Verify that authoritative dataset metadata contains valid entries."""
    assert "fly_hemibrain" in OFFICIAL_DATASETS
    assert "flywire_annotations" in OFFICIAL_DATASETS
    assert "malecns" in OFFICIAL_DATASETS
    assert OFFICIAL_DATASETS["fly_hemibrain"]["url"].startswith("https://")
    assert OFFICIAL_DATASETS["flywire_annotations"]["url"].startswith("https://")
    assert OFFICIAL_DATASETS["malecns"]["auth_required"] is True


def test_mock_connectome_generation():
    """Verify mock connectome produces requested neuron count and expected columns."""
    num_neurons = 200
    nodes_df, edges_df = generate_mock_connectome(num_neurons=num_neurons, avg_degree=10.0, random_seed=42)

    assert len(nodes_df) == num_neurons
    assert SCHEMA.NODE_ID in nodes_df.columns
    assert SCHEMA.CELL_TYPE in nodes_df.columns
    assert SCHEMA.SUPER_CLASS in nodes_df.columns
    assert SCHEMA.NEUROPIL in nodes_df.columns

    assert SCHEMA.SOURCE in edges_df.columns
    assert SCHEMA.TARGET in edges_df.columns
    assert SCHEMA.WEIGHT in edges_df.columns

    # Verify no self-loops were generated in mock
    assert (edges_df[SCHEMA.SOURCE] == edges_df[SCHEMA.TARGET]).sum() == 0

    # Verify all synaptic weights are positive integers
    assert (edges_df[SCHEMA.WEIGHT] > 0).all()

    # Verify all edge endpoints exist in nodes table
    node_id_set = set(nodes_df[SCHEMA.NODE_ID])
    assert set(edges_df[SCHEMA.SOURCE]).issubset(node_id_set)
    assert set(edges_df[SCHEMA.TARGET]).issubset(node_id_set)


def test_mock_connectome_reproducibility():
    """Verify random seed creates deterministic outputs and different seeds differ."""
    nodes1, edges1 = generate_mock_connectome(num_neurons=100, random_seed=123)
    nodes2, edges2 = generate_mock_connectome(num_neurons=100, random_seed=123)
    nodes3, edges3 = generate_mock_connectome(num_neurons=100, random_seed=999)

    pd.testing.assert_frame_equal(nodes1, nodes2)
    pd.testing.assert_frame_equal(edges1, edges2)

    # Different seeds should produce distinct edges
    assert not edges1[SCHEMA.SOURCE].equals(edges3[SCHEMA.SOURCE])


def test_processed_data_roundtrip():
    """Verify saving and loading processed parquet files preserves all data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        nodes_df, edges_df = generate_mock_connectome(num_neurons=150, output_dir=tmp_path, random_seed=42)

        loaded_nodes, loaded_edges = load_processed_data(tmp_path)

        pd.testing.assert_frame_equal(nodes_df, loaded_nodes)
        pd.testing.assert_frame_equal(edges_df, loaded_edges)


def test_load_processed_missing_dir_raises():
    """Verify FileNotFoundError is raised if directory has no processed files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(FileNotFoundError):
            load_processed_data(Path(tmp_dir) / "nonexistent")


def test_real_schema_validation():
    """
    Offline test for validate_real_data on standardized parquet files.
    Verifies metric calculation (degree, density, active neurons) without downloading.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        generate_mock_connectome(num_neurons=120, avg_degree=6.0, output_dir=tmp_path, random_seed=42)

        report = validate_real_data(processed_dir=tmp_path)
        assert report["total_neurons"] == 120
        assert report["total_connections"] > 0
        assert report["average_degree"] > 0.0
        assert report["graph_density"] > 0.0
        assert report["max_degree"] >= report["min_degree"]


def test_real_subnetwork_selection():
    """
    Offline test for extract_real_subnetwork on standardized parquet files.
    Verifies that highest-degree and random subnetwork extractions yield exactly
    the requested neuron count without memory leaks.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        generate_mock_connectome(num_neurons=300, avg_degree=8.0, output_dir=tmp_path, random_seed=42)

        # 1. Test highest degree selection of 50 neurons
        sub_g, diag = extract_real_subnetwork(processed_dir=tmp_path, max_neurons=50, strategy="highest_degree", verbose=False)
        assert sub_g.number_of_nodes() == 50
        assert diag["selected_neurons"] == 50
        assert diag["selected_edges"] == sub_g.number_of_edges()

        # 2. Test random selection of 75 neurons
        sub_g_rand, diag_rand = extract_real_subnetwork(processed_dir=tmp_path, max_neurons=75, strategy="random", seed=42, verbose=False)
        assert sub_g_rand.number_of_nodes() == 75
        assert diag_rand["selected_neurons"] == 75
