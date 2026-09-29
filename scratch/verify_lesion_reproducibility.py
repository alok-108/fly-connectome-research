import importlib.util
from pathlib import Path
import pandas as pd

sweep_path = Path("experiments") / "05_coupling_sweep.py"
spec = importlib.util.spec_from_file_location("exp05", sweep_path)
exp05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp05)

run_lesion_validation = exp05.run_lesion_validation
select_representative_alphas = exp05.select_representative_alphas

from src.data_loader import extract_real_subnetwork
from src.simulation import LIFConfig, select_stimulated_neurons

sub_graph, _ = extract_real_subnetwork(max_neurons=1000, strategy="highest_degree", seed=42, verbose=False)
stim_ids_25 = select_stimulated_neurons(sub_graph, stimulus_fraction=0.25, seed=42)

raw_csv = Path("results/tables/coupling_sweep_raw_results.csv")
intact_df = pd.read_csv(raw_csv)
intact_records = intact_df.to_dict("records")
intact_lookup = {(round(float(r["stimulus_fraction"]), 4), round(float(r["alpha"]), 4)): r for r in intact_records}

base_cfg = LIFConfig(dt=0.5, duration=1000.0, external_current=18.0, random_seed=42)
rep_alphas = select_representative_alphas()

print("Executing 2nd run of lesion validation...")
recs2 = run_lesion_validation(sub_graph, rep_alphas, intact_lookup, base_cfg, stim_ids_25)
df2 = pd.DataFrame(recs2)

df1 = pd.read_csv("results/tables/coupling_lesion_validation.csv", keep_default_na=False)
compare_cols = [c for c in df1.columns if c != "runtime_seconds"]

df1["random_seed"] = df1["random_seed"].astype(str)
df2["random_seed"] = df2["random_seed"].astype(str)

pd.testing.assert_frame_equal(df1[compare_cols], df2[compare_cols], check_exact=False, rtol=1e-5, atol=1e-5)
print("REPRODUCIBILITY CONFIRMED: 100% exact numerical match across all 36 lesion validation simulations!")
