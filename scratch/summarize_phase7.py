import pandas as pd
import numpy as np

# 7B Summary
df_b = pd.read_csv('results/tables/phase7_stimulus_sensitivity.csv')
print("=" * 60)
print("EXPERIMENT 7B: STIMULUS SENSITIVITY SUMMARY")
print("=" * 60)
for alpha in [0.02, 0.03]:
    print(f"\n--- ALPHA = {alpha:.4f} ---")
    for sf in [0.05, 0.10, 0.25]:
        sub = df_b[(df_b['alpha'] == alpha) & (df_b['stimulus_fraction'] == sf)]
        intact = sub[sub['lesion_strategy'] == 'intact'].iloc[0]
        print(f"Stim: {sf*100:>2.0f}% | Intact Rate: {intact['mean_firing_rate_hz']:.2f} Hz | RecrUnstim: {intact['recruited_unstimulated_fraction']*100:.1f}% | PostRate: {intact['post_stim_rate_hz']:.3f} Hz")
        for lf in [0.10, 0.20]:
            hub = sub[(sub['lesion_strategy'] == 'hub') & (sub['lesion_fraction'] == lf)].iloc[0]
            rnd = sub[(sub['lesion_strategy'] == 'random') & (sub['lesion_fraction'] == lf)]
            print(f"  Hub {lf*100:.0f}%: Rob={hub['activity_robustness']:.4f}, Recr={hub['recruited_unstimulated_fraction']*100:.1f}%, PostRate={hub['post_stim_rate_hz']:.3f} Hz, EdgeRet={hub['edge_retention']*100:.1f}%")
            print(f"  Rnd {lf*100:.0f}%: Rob={rnd['activity_robustness'].mean():.4f} +/- {rnd['activity_robustness'].std():.4f}, Recr={rnd['recruited_unstimulated_fraction'].mean()*100:.1f}%, PostRate={rnd['post_stim_rate_hz'].mean():.3f} Hz, EdgeRet={rnd['edge_retention'].mean()*100:.1f}%")

# 7C Summary
df_c = pd.read_csv('results/tables/phase7_lesion_sensitivity.csv')
print("\n" + "=" * 60)
print("EXPERIMENT 7C: LESION FRACTION SENSITIVITY SUMMARY")
print("=" * 60)
for alpha in [0.02, 0.03]:
    print(f"\n--- ALPHA = {alpha:.4f} ---")
    sub = df_c[df_c['alpha'] == alpha]
    intact = sub[sub['lesion_strategy'] == 'intact'].iloc[0]
    print(f"Intact: Rate={intact['mean_firing_rate_hz']:.2f} Hz, Recr={intact['recruited_unstimulated_fraction']*100:.1f}%, PostRate={intact['post_stim_rate_hz']:.3f} Hz")
    for lf in [0.05, 0.10, 0.20]:
        hub = sub[(sub['lesion_strategy'] == 'hub') & (sub['lesion_fraction'] == lf)].iloc[0]
        rnd = sub[(sub['lesion_strategy'] == 'random') & (sub['lesion_fraction'] == lf)]
        print(f"  Lesion {lf*100:>2.0f}% | Hub Rob: {hub['activity_robustness']:.4f}, Recr: {hub['recruited_unstimulated_fraction']*100:.1f}% | Rnd Rob: {rnd['activity_robustness'].mean():.4f} +/- {rnd['activity_robustness'].std():.4f}, Recr: {rnd['recruited_unstimulated_fraction'].mean()*100:.1f}%")

# 7D Summary
df_d = pd.read_csv('results/tables/phase7_timestep_sensitivity.csv')
print("\n" + "=" * 60)
print("EXPERIMENT 7D: TIMESTEP SENSITIVITY SUMMARY")
print("=" * 60)
for alpha in [0.02, 0.03]:
    print(f"\n--- ALPHA = {alpha:.4f} ---")
    sub = df_d[df_d['alpha'] == alpha]
    for dt in [0.25, 0.50, 1.00]:
        sub_dt = sub[sub['dt'] == dt]
        intact = sub_dt[sub_dt['lesion_strategy'] == 'intact'].iloc[0]
        h10 = sub_dt[(sub_dt['lesion_strategy'] == 'hub') & (sub_dt['lesion_fraction'] == 0.10)].iloc[0]
        h20 = sub_dt[(sub_dt['lesion_strategy'] == 'hub') & (sub_dt['lesion_fraction'] == 0.20)].iloc[0]
        r20 = sub_dt[(sub_dt['lesion_strategy'] == 'random') & (sub_dt['lesion_fraction'] == 0.20)].iloc[0]
        print(f"dt={dt:.2f} ms | Intact: {intact['mean_firing_rate_hz']:.2f} Hz | Hub10 Rob: {h10['activity_robustness']:.4f} | Hub20 Rob: {h20['activity_robustness']:.4f} | Rnd20 Rob: {r20['activity_robustness']:.4f}")

# 7E Summary
df_e = pd.read_csv('results/tables/phase7_subnetwork_sensitivity.csv')
print("\n" + "=" * 60)
print("EXPERIMENT 7E: SUBNETWORK SELECTION SENSITIVITY SUMMARY")
print("=" * 60)
for net in ['highest_degree', 'random']:
    print(f"\n--- NETWORK: {net} ---")
    sub = df_e[df_e['network_selection'] == net]
    for alpha in [0.02, 0.03]:
        sub_a = sub[sub['alpha'] == alpha]
        intact = sub_a[sub_a['lesion_strategy'] == 'intact'].iloc[0]
        h20 = sub_a[(sub_a['lesion_strategy'] == 'hub') & (sub_a['lesion_fraction'] == 0.20)].iloc[0]
        r20 = sub_a[(sub_a['lesion_strategy'] == 'random') & (sub_a['lesion_fraction'] == 0.20)].iloc[0]
        print(f"Alpha={alpha:.4f} | Intact: {intact['mean_firing_rate_hz']:.2f} Hz, Recr={intact['recruited_unstimulated_fraction']*100:.1f}%, PostRate={intact['post_stim_rate_hz']:.3f} Hz")
        print(f"  Hub 20%: Rob={h20['activity_robustness']:.4f}, Rate={h20['mean_firing_rate_hz']:.2f} Hz, EdgeRet={h20['edge_retention']*100:.1f}%")
        print(f"  Rnd 20%: Rob={r20['activity_robustness']:.4f}, Rate={r20['mean_firing_rate_hz']:.2f} Hz, EdgeRet={r20['edge_retention']*100:.1f}%")
