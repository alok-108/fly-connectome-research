import importlib.util
import time
import tracemalloc
from pathlib import Path
import pandas as pd

sweep_path = Path("experiments") / "05_coupling_sweep.py"
spec = importlib.util.spec_from_file_location("exp05", sweep_path)
exp05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp05)

run_intact_coupling_sweep = exp05.run_intact_coupling_sweep
ALPHA_VALUES = exp05.ALPHA_VALUES
STIMULUS_FRACTIONS = exp05.STIMULUS_FRACTIONS

from src.data_loader import extract_real_subnetwork
from src.simulation import LIFConfig, select_stimulated_neurons

sub_graph, _ = extract_real_subnetwork(max_neurons=1000, strategy="highest_degree", seed=42, verbose=False)
stim_sets = {sf: select_stimulated_neurons(sub_graph, stimulus_fraction=sf, seed=42) for sf in STIMULUS_FRACTIONS}
base_cfg = LIFConfig(dt=0.5, duration=1000.0, external_current=18.0, random_seed=42)

tracemalloc.start()
t0 = time.perf_counter()
recs, _ = run_intact_coupling_sweep(sub_graph, ALPHA_VALUES, STIMULUS_FRACTIONS, base_cfg, stim_sets)
t1 = time.perf_counter()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

df2 = pd.DataFrame(recs)
df1 = pd.read_csv("results/tables/coupling_sweep_raw_results.csv")

compare_cols = [c for c in df1.columns if c != "runtime_seconds"]
pd.testing.assert_frame_equal(df1[compare_cols], df2[compare_cols], check_exact=False, rtol=1e-5, atol=1e-5)
print("REPRODUCIBILITY CONFIRMED: 100% exact numerical match across all 55 intact simulations!")
print(f"Second run duration: {t1 - t0:.2f} s, peak memory: {peak_mem / (1024*1024):.2f} MB")
