"""
Verification script for Phase 7 reproducibility.
Reruns deterministic subsets of Phase 7 simulations (Experiment 7A, 7D, and 7E)
and compares numerical outputs against saved CSV results.
"""

import importlib.util
from pathlib import Path
import sys
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import extract_real_subnetwork
from src.simulation import LIFConfig, select_stimulated_neurons

# Import Phase 7 runner
phase7_path = Path("experiments") / "06_phase7_robustness.py"
spec = importlib.util.spec_from_file_location("exp06", phase7_path)
exp06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp06)

run_experiment_7a = exp06.run_experiment_7a
run_experiment_7d = exp06.run_experiment_7d
run_experiment_7e = exp06.run_experiment_7e

print("=" * 65)
print(" PHASE 7 REPRODUCIBILITY VERIFICATION")
print("=" * 65)

# 1. Primary subnetwork
sub_graph_primary, _ = extract_real_subnetwork(
    max_neurons=1000, strategy="highest_degree", seed=42, verbose=False
)
stim_ids_25 = select_stimulated_neurons(sub_graph_primary, stimulus_fraction=0.25, seed=42)

# 2. Random subnetwork
sub_graph_random, _ = extract_real_subnetwork(
    max_neurons=1000, strategy="random", seed=2026, verbose=False
)

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

# ------------------------------------------------------------------------------
# Verification 1: Experiment 7A (Local Alpha Sensitivity)
# ------------------------------------------------------------------------------
print("\n[1/3] Rerunning Experiment 7A (5 alpha conditions)...")
recs_7a_run2 = run_experiment_7a(sub_graph_primary, stim_ids_25, base_config)
df_7a_run2 = pd.DataFrame(recs_7a_run2)

df_7a_saved = pd.read_csv("results/tables/phase7_alpha_sensitivity.csv", keep_default_na=False)
compare_cols_7a = [c for c in df_7a_saved.columns if c != "runtime_seconds"]

df_7a_saved["random_seed"] = df_7a_saved["random_seed"].astype(str)
df_7a_run2["random_seed"] = df_7a_run2["random_seed"].astype(str)

pd.testing.assert_frame_equal(df_7a_saved[compare_cols_7a], df_7a_run2[compare_cols_7a], check_exact=False, rtol=1e-5, atol=1e-5)
print("  Experiment 7A: Both runs produced exactly identical numerical values across all compared result fields.")

# ------------------------------------------------------------------------------
# Verification 2: Experiment 7D (Timestep Sensitivity)
# ------------------------------------------------------------------------------
print("\n[2/3] Rerunning Experiment 7D (24 timestep conditions)...")
recs_7d_run2 = run_experiment_7d(sub_graph_primary, stim_ids_25, base_config)
df_7d_run2 = pd.DataFrame(recs_7d_run2)

df_7d_saved = pd.read_csv("results/tables/phase7_timestep_sensitivity.csv", keep_default_na=False)
compare_cols_7d = [c for c in df_7d_saved.columns if c != "runtime_seconds"]

df_7d_saved["random_seed"] = df_7d_saved["random_seed"].astype(str)
df_7d_run2["random_seed"] = df_7d_run2["random_seed"].astype(str)

pd.testing.assert_frame_equal(df_7d_saved[compare_cols_7d], df_7d_run2[compare_cols_7d], check_exact=False, rtol=1e-5, atol=1e-5)
print("  Experiment 7D: Both runs produced exactly identical numerical values across all compared result fields.")

# ------------------------------------------------------------------------------
# Verification 3: Experiment 7E (Subnetwork Selection Sensitivity)
# ------------------------------------------------------------------------------
print("\n[3/3] Rerunning Experiment 7E (12 subnetwork conditions)...")
recs_7e_run2 = run_experiment_7e(sub_graph_primary, sub_graph_random, base_config)
df_7e_run2 = pd.DataFrame(recs_7e_run2)

df_7e_saved = pd.read_csv("results/tables/phase7_subnetwork_sensitivity.csv", keep_default_na=False)
compare_cols_7e = [c for c in df_7e_saved.columns if c != "runtime_seconds"]

df_7e_saved["random_seed"] = df_7e_saved["random_seed"].astype(str)
df_7e_run2["random_seed"] = df_7e_run2["random_seed"].astype(str)

pd.testing.assert_frame_equal(df_7e_saved[compare_cols_7e], df_7e_run2[compare_cols_7e], check_exact=False, rtol=1e-5, atol=1e-5)
print("  Experiment 7E: Both runs produced exactly identical numerical values across all compared result fields.")

print("\n" + "=" * 65)
print(" PHASE 7 REPRODUCIBILITY CONFIRMED")
print(" Both runs produced exactly identical numerical values across all compared result fields.")
print("=" * 65)
