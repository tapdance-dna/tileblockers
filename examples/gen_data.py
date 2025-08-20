from tqdm import tqdm
from tileblockers.twelve_helix_tube import rate_per_hour_sim, run_ffs_for_system, rate_per_hour_sim_with_melting, simple_twelve_helix_system
import polars as pl
import numpy as np
from tqdm.contrib.concurrent import thread_map


bconc = 2.5e-6
temps = np.arange(30, 55, 1)
tile_concs = np.logspace(1, 3, 20) * 1e-9

rows = []
for t in tqdm(temps, desc="Temps"):
    for b in tile_concs:
        gr = rate_per_hour_sim_with_melting(
            t, bconc/b, n_sims=12, sys_fun=simple_twelve_helix_system, tile_conc = b, tile_remaining=1.0
            ) / 3600.0 / 3.5
        rows.append({"temperature": t, "tile_conc": b, "growth_rate": gr, 'blocker_mult': bconc/b, 'blocker_conc': bconc})

vals = pl.DataFrame(rows)

vals.write_csv("phase_diagram_data_sims_growth.csv")



params = [dict(temp=x['temperature'], cov_mult=x['blocker_mult'], tile_conc=x['tile_conc'], var_per_mean2=0.01, min_nuc_rate=1e-14, 
    sys_fun=simple_twelve_helix_system) for x in vals.iter_rows(named=True)]
ffs_res = thread_map(lambda args: run_ffs_for_system(**args), params)
vals = vals.with_columns(nucleation_rate=pl.Series([x[0] for x in ffs_res]), )

vals.write_csv("phase_diagram_data_sims_growth.csv")

