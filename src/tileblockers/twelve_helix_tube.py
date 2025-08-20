from typing import Any, Callable
import rgrow as rg
import numpy as np
from rgrow.kblock import KBlockTile, KBlockParams
import polars as pl
from rgrow import KBlock

from .theoretical_calculations import growth_rate


from .constants import DS_LAT, R_CONST, TILE_CONC, SINGLE_SEQ, COLORSET

# "k=10" sticky end sequences, in NESW order, from Rogers et al.
TILE_GLUE_SEQUENCES_K10 = [
    ["GGAACAGACT", "GGTTCTTCTC", "GGTCTAGTCG", "TTGAACTTGG"],
    ["CGACTAGACC", "CCAAGTTCAA", "CGTCAATACC", "TGAACAGACA"],
    ["GGTATTGACG", "TGTCTGTTCA", "TCGTCTCTTG", "GCAAGATTGA"],
    ["CAAGAGACGA", "TCAATCTTGC", "AATTCTGTCG", "GCAAACAGAA"],
    ["CGACAGAATT", "TTCTGTTTGC", "GTCTTGTTCA", "GGTAGATTCG"],
    ["TGAACAAGAC", "CGAATCTACC", "GCTCTAGTCT", "GCATTGAACC"],
    ["AGACTAGAGC", "GGTTCAATGC", "TCTGTGTTCA", "TGACAAGACA"],
    ["TGAACACAGA", "TGTCTTGTCA", "GTTCCAGTCT", "CGAACAAAGG"],
    ["AGACTGGAAC", "CCTTTGTTCG", "CGTCTCAGTT", "ACAGACAGAC"],
    ["AACTGAGACG", "GTCTGTCTGT", "GGTCTGAATG", "CGACTTCTTC"],
    ["CATTCAGACC", "GAAGAAGTCG", "CACAGAGAGT", "GGACAATACG"],
    ["ACTCTCTGTG", "CGTATTGTCC", "AGTCTGTTCC", "GAGAAGAACC"],
]


# tns = range(1, len(sys.tile_names))
# dvs = []
# bvs = []
# RTA = R_CONST * (sys.temperature + 273.15) * sys.alpha
# for tn in tns:
#     n,e,s,w = sys.py_get_tile_uncovered_glues(tn << 4)
#     dvs.append(sys.glue_links[n-1,n] + sys.glue_links[w-1,w] + RTA)
#     bvs.append(sys.glue_links[n-1,n] + RTA)
#     bvs.append(sys.glue_links[w-1,w] + RTA)

# plt.plot(tns, dvs)
# plt.show()



def theoretical_growth_rate_over_temp_for_sysfunc(sysfunc: Callable[[float, float], rg.KBlock], 
                                                  cov_mult: float, tile_conc: float = TILE_CONC, 
                                                  temps=np.linspace(20, 60, 100), order=2,
                                                  use_percentile: None | float = None):
    if not use_percentile:
        gvals = []
        for temp in temps:
            sys = sysfunc(temp, 10)
            gvals.append(sys.glue_links[sys.glue_links.nonzero()].mean())  #  + sys.alpha * R_CONST * (273.15 + sys.temperature)
        gvals = np.array(gvals)
        ratevals = growth_rate(temps, cov_mult, tile_conc, dg=gvals, order=order)
    if use_percentile is not None:
        gvals = []
        tbgvals = []
        for temp in temps:
            sys = sysfunc(temp, 10)
            # rta = R_CONST * (sys.temperature + 273.15) * sys.alpha
            dvs = []
            bvs = []
            for tn in range(1, len(sys.tile_names)):
                n,e,s,w = sys.py_get_tile_uncovered_glues(tn << 4)
                dvs.append(sys.glue_links[n-1,n] + sys.glue_links[w-1,w] - (273.15 + sys.temperature) * sys.ds_lat) # + rta)
                bvs.append(sys.glue_links[n-1,n])  # + rta)
                bvs.append(sys.glue_links[w-1,w])  # + rta)
            tbdg = np.percentile(dvs, use_percentile)
            dg = np.percentile(bvs, 100-use_percentile)
            gvals.append(dg)
            tbgvals.append(tbdg)
        ratevals = growth_rate(temps, cov_mult, tile_conc, dg=np.array(gvals), tbdg=np.array(tbgvals), order=order)
                
    return ratevals

def simple_twelve_helix_system(
    temp: float,
    cov_mult: float,
    tile_conc: float = TILE_CONC,
    tile_remaining: float = 1.0,
    diag: bool = False,
    kblockparams: dict[str, Any] | None = None
) -> rg.System:
    if kblockparams is None:
        kblockparams = {}
    tiles = [
        KBlockTile(
            f"tile_{i}",
            tile_conc * tile_remaining,
            [f"GN_{i}", f"GE_{i}", f"GN_{(i + 1) % 12}*", f"GE_{(i + 1) % 12}*"],
            color = COLORSET[i]
        )
        for i in range(12)
    ]

    if diag:
        seed = {
            (0, 20): ("tile_1"),
            (2, 19): ("tile_3"),
            (4, 18): ("tile_5"),
            (6, 17): ("tile_7"),
            (8, 16): ("tile_9"),
            (10, 15): ("tile_11"),
        }
    else:
        seed = {
            (0, 2): ("tile_1"),
            (2, 2): ("tile_3"),
            (4, 2): ("tile_5"),
            (6, 2): ("tile_7"),
            (8, 2): ("tile_9"),
            (10, 2): ("tile_11"),
        }
    
    def blocker_conc_list(
        blocker_mult: float, tile_conc: float = TILE_CONC
    ) -> dict[str, float]:
        return {f"GN_{i}": blocker_mult * tile_conc for i in range(12)} | {
            f"GE_{i}": blocker_mult * tile_conc for i in range(12)
        }

    glue_names = [f"GN_{i}" for i in range(12)] + [f"GE_{i}" for i in range(12)]
    glue_seqs = [SINGLE_SEQ for i in range(12)] + [
        SINGLE_SEQ for i in range(12)
    ]
    binding_strength = dict(zip(glue_names, glue_seqs))
    params = KBlockParams(
        tiles, blocker_conc_list(cov_mult, tile_conc), seed, binding_strength, 
        temp=temp, **kblockparams
    )
    return KBlock(params)

def rate_per_hour_sim_tosize(
    temp,
    cov_mult,
    sys_fun = simple_twelve_helix_system,
    gseq = SINGLE_SEQ,
    n_sims: int = 100,
    tile_conc: float = TILE_CONC,
    tile_remaining: float = 1.0,
    time_to_run: float = 36000,
    to_size: int = 100*12,
    kblockparams: dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kblockparams is None:
        kblockparams = {}
    sys = sys_fun(temp, cov_mult, tile_conc, tile_remaining, kblockparams=kblockparams)

    states = [new_state(sys, 256) for _ in range(n_sims)]
    sys.evolve(states, for_time=time_to_run, size_max=to_size)
    ntiles = np.array([state.ntiles for state in states]) - len(sys.seed)
    times = np.array([state.time for state in states])
    return 3600 * ntiles / times, ntiles, times

import scipy.stats as stats
def pctile_nr(fr, p):
    d = fr.surfaces_dataframe()
    return fr.dimerization_rate * np.prod(stats.binom.ppf(p, d['n_trials'], d['p_r'])/d['n_trials'].to_numpy())

def run_ffs_for_system(temp, cov_mult, sys_fun = simple_twelve_helix_system, kblockparams: dict[str, Any] | None = None, 
                       tile_conc: float = TILE_CONC, tile_remaining: float = 1.0, min_nuc_rate: float = 1e-24, 
                       var_per_mean2: float = 1e-2, **ffs_kwargs):
    if kblockparams is None:
        kblockparams = {}
    sys = sys_fun(temp, cov_mult, tile_conc, tile_remaining, kblockparams=kblockparams)
    result = sys.run_ffs(min_nuc_rate=min_nuc_rate, var_per_mean2=var_per_mean2, **ffs_kwargs)
    # Calculate percentiles once to avoid redundant calculations
    nr = result.nucleation_rate
    p05 = pctile_nr(result, 0.05)
    p95 = pctile_nr(result, 0.95)
    # print(f"{temp:.2f},{nr:.2e},{p05:.2e},{p95:.2e}", flush=True)
    return nr, p05, p95, result

def twelve_helix_system(
    temp: float,
    cov_mult: float,
    tile_conc: float = TILE_CONC,
    tile_remaining: float = 1.0,
    diag: bool = False,
    kblockparams: dict[str, Any] | None = None
) -> rg.System:
    if kblockparams is None:
        kblockparams = {}
    tiles = [
        KBlockTile(
            f"tile_{i}",
            tile_conc * tile_remaining,
            [f"GN_{i}", f"GE_{i}", f"GN_{(i + 1) % 12}*", f"GE_{(i + 1) % 12}*"],
            color = COLORSET[i]
        )
        for i in range(12)
    ]

    if diag:
        seed = {
            (0, 20): ("tile_1"),
            (2, 19): ("tile_3"),
            (4, 18): ("tile_5"),
            (6, 17): ("tile_7"),
            (8, 16): ("tile_9"),
            (10, 15): ("tile_11"),
        }
    else:
        seed = {
            (0, 2): ("tile_1"),
            (2, 2): ("tile_3"),
            (4, 2): ("tile_5"),
            (6, 2): ("tile_7"),
            (8, 2): ("tile_9"),
            (10, 2): ("tile_11"),
        }
    
    def blocker_conc_list(
        blocker_conc: float, tile_conc: float = TILE_CONC
    ) -> dict[str, float]:
        return {f"GN_{i}": blocker_conc * tile_conc for i in range(12)} | {
            f"GE_{i}": blocker_conc * tile_conc for i in range(12)
        }

    glue_names = [f"GN_{i}" for i in range(12)] + [f"GE_{i}" for i in range(12)]
    glue_seqs = [TILE_GLUE_SEQUENCES_K10[i][0] for i in range(12)] + [
        TILE_GLUE_SEQUENCES_K10[i][1] for i in range(12)
    ]
    binding_strength = dict(zip(glue_names, glue_seqs))
    params = KBlockParams(
        tiles, blocker_conc_list(cov_mult, tile_conc), seed, binding_strength, temp=temp, **kblockparams
    )
    return KBlock(params)

def k9_system(
    temp: float,
    cov_mult: float,
    tile_conc: float = TILE_CONC,
    tile_remaining: float = 1.0,
    kblockparams: dict[str, Any] | None = None
) -> rg.KBlock:
    if kblockparams is None:
        kblockparams = {}
    T9_ALL_SEQS = pl.read_csv("experimental-data/sequences-9-no-nonrep.csv")

    tsd = T9_ALL_SEQS.with_columns(pl.col("Sequence").str.split(" ").alias("Sequence")).with_columns(
        pl.col("Sequence").list.get(2).alias("N"),
        pl.col("Sequence").list.get(5).alias("S"),
        pl.col("Sequence").list.get(0).alias("E"),
        pl.col("Sequence").list.get(3).alias("W"),
    )

    def revcomp(seq: str) -> str:
        return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))

    def revcomps(seq: str) -> str:
        return revcomp(seq) + "*"

    tsd = tsd.with_columns(
        pl.col("N").map_elements(revcomps, return_dtype=pl.Utf8).alias("N"),
        pl.col("W").map_elements(revcomps, return_dtype=pl.Utf8).alias("W"),
    ).drop("Sequence")

    blocker_conc = cov_mult * tile_conc
    cover_dict = {x: blocker_conc for x in tsd["S"].to_list() + tsd["E"].to_list()}

    bonds = {x: x for x in tsd["S"].to_list() + tsd["E"].to_list()}

    tiles = [
        KBlockTile(name=d["Name"].replace("Tile_",""), glues=[d[x] for x in ["N", "E", "S", "W"]], concentration=tile_conc * tile_remaining)
        for d in tsd.iter_rows(named=True)
    ]

    seed = {
        (0, 2): "99_5",
        (2, 2): "99_0",
        (4, 2): "99_1",
        (6, 2): "99_2",
        (8, 2): "99_3",
        (10, 2): "99_4",
    }

    params = KBlockParams(tiles=tiles, blocker_conc=cover_dict, seed=seed, binding_strength=bonds, temp=temp, **kblockparams)

    return KBlock(params)

def k10_system(
    temp: float,
    cov_mult: float,
    tile_conc: float = TILE_CONC,
    tile_remaining: float = 1.0,
    kblockparams: dict[str, Any] | None = None
) -> rg.KBlock:
    if kblockparams is None:
        kblockparams = {}
    T9_ALL_SEQS = pl.read_csv("experimental-data/seqs-10.csv")

    tsd = T9_ALL_SEQS.with_columns(pl.col("Sequence").str.split(" ").alias("Sequence")).with_columns(
        pl.col("Sequence").list.get(2).alias("N"),
        pl.col("Sequence").list.get(5).alias("S"),
        pl.col("Sequence").list.get(0).alias("E"),
        pl.col("Sequence").list.get(3).alias("W"),
    )

    def revcomp(seq: str) -> str:
        return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))

    def revcomps(seq: str) -> str:
        return revcomp(seq) + "*"

    tsd = tsd.with_columns(
        pl.col("N").map_elements(revcomps, return_dtype=pl.Utf8).alias("N"),
        pl.col("W").map_elements(revcomps, return_dtype=pl.Utf8).alias("W"),
    ).drop("Sequence")

    blocker_conc = cov_mult * tile_conc
    cover_dict = {x: blocker_conc for x in tsd["S"].to_list() + tsd["E"].to_list()}

    bonds = {x: x for x in tsd["S"].to_list() + tsd["E"].to_list()}

    tiles = [
        KBlockTile(name=d["Name"].replace("Tile_", ""), glues=[d[x] for x in ["N", "E", "S", "W"]], concentration=tile_conc * tile_remaining)
        for d in tsd.iter_rows(named=True)
    ]

    # # Find tiles with names 19_[1-5] and update their N and W glues with E and S glues from 8_[1-5]
    # for i in range(0, 6):
    #     # Find the corresponding tiles
    #     tile_19 = next((t for t in tiles if t.name == f"19_{i}"), None)
    #     tile_8_w = next((t for t in tiles if t.name == f"8_{(i+1)%6}"), None)
    #     tile_8_n = next((t for t in tiles if t.name == f"8_{i}"), None)
    #     if tile_19 and tile_8_w and tile_8_n:
    #         # Get E and S glues from tile_8
    #         tile_19.glues[3] = (tile_8_w.glues[1]) + "*"
    #         tile_19.glues[0] = (tile_8_n.glues[2]) + "*"
            

    seed = {
        (0, 2): "1019_5",
        (2, 2): "1019_0",
        (4, 2): "1019_1",
        (6, 2): "1019_2",
        (8, 2): "1019_3",
        (10, 2): "1019_4",
    }

    params = KBlockParams(tiles=tiles, blocker_conc=cover_dict, seed=seed, binding_strength=bonds, temp=temp, **kblockparams)

    return KBlock(params)


def new_state(system: rg.System, length: int = 256, diag: bool = False) -> rg.State:
    st = rg.State((12, length), "tube-diag" if diag else "tube", "none")
    system.setup_state(st)
    return st


def rate_per_hour_sim(
    temp,
    cov_mult,
    n_sims: int = 100,
    init_length: int = 256,
    tile_conc: float = TILE_CONC,
    tile_remaining: float = 1.0,
    sys_fun = twelve_helix_system,
    time_to_run: float = 3600,
    kblockparams: dict[str, Any] | None = None
) -> float:
    if kblockparams is None:
        kblockparams = {}
    sys = sys_fun(temp, cov_mult, tile_conc, tile_remaining, **kblockparams)

    length = init_length
    max_tiles = 12 * (length - 24)  # to be safe
    while True:
        states = [new_state(sys, length) for _ in range(n_sims)]
        sys.evolve(states, for_time=time_to_run)
        ntiles = np.array([state.ntiles for state in states])
        if ntiles.max() < max_tiles:
            break
        length *= 2
        max_tiles = 12 * (length - 8)
    return ((ntiles - len(sys.seed)).mean() / time_to_run) * 3600


def rate_per_hour_sim_with_melting(
    temp,
    cov_mult,
    n_sims: int = 100,
    length: int = 256,
    tile_conc: float = TILE_CONC,
    tile_remaining: float = 1.0,
    sys_fun = twelve_helix_system,
    kblockparams: dict[str, Any] | None = None,
    safe_growth_temp: float = 46.0,
    start_size: int = 128*12,
    timeout = 10*3600
) -> float:
    if kblockparams is None:
        kblockparams = {}
    sys = sys_fun(safe_growth_temp, 0.0, 1e-7, 1.0, **kblockparams)

    max_tiles = 12 * (length - 24)  # to be safe
    ex_state = new_state(sys, length)
    min_tiles = ex_state.ntiles + 24

    states = [new_state(sys, length) for _ in range(n_sims)]
    sys.evolve(states, size_max=start_size, for_time=timeout)

    times = np.array([state.time for state in states])
    ntiles = np.array([state.ntiles for state in states])

    sys = sys_fun(temp, cov_mult, tile_conc, tile_remaining, **kblockparams)
    for x in states:
        sys.update_state(x)
    
    sys.evolve(states, size_max=max_tiles, size_min=min_tiles, for_time=timeout)

    times_after = np.array([state.time for state in states])
    ntiles_after = np.array([state.ntiles for state in states])


    return ((ntiles_after - ntiles) / (times_after - times)).mean() * 3600




def dataline(
    temp: float,
    cov_mult: float,
    n_growth_sims: int = 100,
    run_ffs: bool = False,
    init_growth_length: int = 1024,
    tile_conc: float = TILE_CONC,
    tile_remaining: float = 1.0,
    sys_fun = k10_system,
    time_to_run: float = 3600,
    output_file: str = None,
    ffs_kwargs: dict[str, Any] | None = None,
    kblockparams: dict[str, Any] | None = None
) -> dict:
    """
    Run simulation with specified parameters and return results as a dictionary that can be serialized to NDJSON.
    If output_file is provided, results will be appended to that file in NDJSON format.
    """
    import datetime
    import socket
    import orjson

    if ffs_kwargs is None:
        ffs_kwargs = {}
    if kblockparams is None:
        kblockparams = {}
    default_ffs_kwargs = dict(
        canvas_size=(12, 64),
        canvas_type="tube",
        var_per_mean2=0.001,
        target_size=50,
        min_nuc_rate=1e-14,
    )
    ffs_kwargs = default_ffs_kwargs | ffs_kwargs
    sys = sys_fun(temp, cov_mult, tile_conc, tile_remaining, kblockparams=kblockparams)
    
    rates, sizes, times = rate_per_hour_sim_tosize(
        temp, cov_mult, sys_fun, n_sims=n_growth_sims, tile_conc=tile_conc, 
        tile_remaining=tile_remaining, time_to_run=time_to_run, to_size=init_growth_length, kblockparams=kblockparams
    )
    
    # Calculate statistics
    mean_rate = float(rates.mean())
    rate_5p, rate_95p = float(np.percentile(rates, [5, 95])[0]), float(np.percentile(rates, [5, 95])[1])
    
    # Create result dictionary
    result = {
        # Simulation parameters
        'temperature': float(temp),
        'blocker_multiple': float(cov_mult),
        'system_type': sys_fun.__name__,
        'n_sims': n_growth_sims,
        'tile_concentration': float(tile_conc),
        'tile_remaining': float(tile_remaining),
        'time_to_run': float(time_to_run),
        'to_size': int(init_growth_length),
        # Metadata
        'timestamp': datetime.datetime.now().isoformat(),
        'hostname': socket.gethostname(),
        # Results
        'rate_mean': mean_rate,
        'rate_5percentile': rate_5p,
        'rate_95percentile': rate_95p,
        'rates': rates.tolist(),
        'sizes': sizes.tolist(),
        'times': times.tolist(),
        **kblockparams
    }
    
    # Add FFS data if requested
    if run_ffs:
        ffs_rate = sys.run_ffs(**ffs_kwargs).nucleation_rate
        result['ffs_rate'] = float(ffs_rate)
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'ab') as f:
            f.write(orjson.dumps(result) + b'\n')
    
    return result

