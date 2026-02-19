#!/usr/bin/env python3
"""
Generate phase diagram data with growth and nucleation rates.
Supports command line arguments for parameter ranges and outputs results line-by-line.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from tileblockers.twelve_helix_tube import (
    rate_per_hour_sim_with_melting, 
    run_ffs_for_system, 
    simple_twelve_helix_system,
    twelve_helix_system,
    k9_system,
    k10_system
)
from tileblockers.constants import TILE_CONC
import polars as pl
import numpy as np


def parse_parameter(param_str):
    """
    Parse a parameter string that can be either:
    - A single value: "2.5e-6"
    - A range: "30:55:0.5" (start:stop:step)
    - A logspace range: "log:1:3:20" (log:start:stop:num_points)
    - A list of values: "1.0,2.5,5.0" (comma-separated, processed in order)
    """
    if param_str is None:
        return None
    
    if param_str.startswith("log:"):
        parts = param_str.split(":")
        if len(parts) != 4:
            raise ValueError(f"Logspace format should be 'log:start:stop:num_points', got {param_str}")
        _, start, stop, num = parts
        return np.logspace(float(start), float(stop), int(num))
    elif "," in param_str:
        # Parse comma-separated list of values
        values = [float(val.strip()) for val in param_str.split(",")]
        return np.array(values)
    elif ":" in param_str:
        parts = param_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"Range format should be 'start:stop:step', got {param_str}")
        start, stop, step = map(float, parts)
        return np.arange(start, stop, step)
    else:
        return [float(param_str)]


def generate_filename(temps, tile_concs, bconcs=None, bmults=None):
    """Generate filename based on parameter ranges"""
    def format_range(values, name):
        if len(values) == 1:
            return f"{name}_{values[0]:.2e}"
        else:
            return f"{name}_{values[0]:.2e}_to_{values[-1]:.2e}_n{len(values)}"
    
    temp_str = format_range(temps, "T")
    tile_str = format_range(tile_concs, "tile")
    
    if bmults is not None:
        blocker_str = format_range(bmults, "bmult")
    else:
        blocker_str = format_range(bconcs, "bconc")
    
    return f"phase_diagram_data_{temp_str}_{tile_str}_{blocker_str}.csv"


def create_parameter_info(temps, tile_concs, bconcs=None, bmults=None, n_sims=None, var_per_mean2=None, args=None, loop_order=None):
    """Create parameter information dictionary for JSON output"""
    blocker_count = len(bmults) if bmults is not None else len(bconcs)
    blocker_param_name = 'bmults' if bmults is not None else 'bconcs'
    default_loop_order = ['temps', 'tile_concs', blocker_param_name]
    
    return {
        "generation_info": {
            "timestamp": datetime.now().isoformat(),
            "script_version": "tileblockers-gen-data",
            "total_simulations": len(temps) * len(tile_concs) * blocker_count,
            "loop_order": loop_order or default_loop_order,
            "loop_order_note": "Parameters are nested in this order (first = outer loop, last = inner loop)"
        },
        "simulation_parameters": {
            "n_sims_per_point": n_sims,
            "var_per_mean2": var_per_mean2,
            "max_sim_time": args.max_sim_time,
            "start_size": args.start_size,
            "length": args.length,
            "sys_fun": args.sys_fun
        },
        "parameter_ranges": {
            "temperatures": {
                "values": temps.tolist() if hasattr(temps, 'tolist') else list(temps),
                "unit": "°C",
                "count": len(temps),
                "range": f"{temps[0]:.2f} to {temps[-1]:.2f}" if len(temps) > 1 else f"{temps[0]:.2f}",
                "original_spec": args.temps
            },
            "tile_concentrations": {
                "values": tile_concs.tolist() if hasattr(tile_concs, 'tolist') else list(tile_concs),
                "unit": "M",
                "count": len(tile_concs),
                "range": f"{tile_concs[0]:.2e} to {tile_concs[-1]:.2e}" if len(tile_concs) > 1 else f"{tile_concs[0]:.2e}",
                "original_spec": args.tile_concs,
                "note": "Values are multiplied by 1e-9 from input specification"
            },
            "blocker_concentrations": {
                "values": bconcs.tolist() if hasattr(bconcs, 'tolist') else list(bconcs) if bconcs is not None else None,
                "unit": "M",
                "count": len(bconcs) if bconcs is not None else None,
                "range": f"{bconcs[0]:.2e} to {bconcs[-1]:.2e}" if bconcs is not None and len(bconcs) > 1 else f"{bconcs[0]:.2e}" if bconcs is not None else None,
                "original_spec": getattr(args, 'bconcs', None) if args else None
            },
            "blocker_multipliers": {
                "values": bmults.tolist() if hasattr(bmults, 'tolist') else list(bmults) if bmults is not None else None,
                "unit": "dimensionless (ratio)",
                "count": len(bmults) if bmults is not None else None,
                "range": f"{bmults[0]:.2e} to {bmults[-1]:.2e}" if bmults is not None and len(bmults) > 1 else f"{bmults[0]:.2e}" if bmults is not None else None,
                "original_spec": getattr(args, 'bmults', None) if args else None,
                "note": "Ratio of blocker concentration to tile concentration" if bmults is not None else None
            }
        },
        "output_info": {
            "csv_columns": [
                "temperature", "tile_conc", "blocker_conc", "blocker_mult", 
                "growth_rate", "nucleation_rate", "nucleation_rate_05", "nucleation_rate_95"
            ]
        }
    }


def generate_parameter_combinations(params_dict, specified_order):
    """Generate parameter combinations in nested loop order based on specification order"""
    # Default order for unspecified parameters
    blocker_param = 'bmults' if 'bmults' in params_dict else 'bconcs'
    default_order = ['temps', 'tile_concs', blocker_param]
    
    # Create ordered parameter list: specified parameters first, then unspecified ones
    ordered_params = []
    for param in specified_order:
        if param in params_dict:
            ordered_params.append(param)
    
    # Add any unspecified parameters at the end
    for param in default_order:
        if param not in ordered_params:
            ordered_params.append(param)
    
    # Create nested iteration
    def recursive_combinations(param_index, current_combination):
        if param_index >= len(ordered_params):
            yield current_combination.copy()
            return
        
        param_name = ordered_params[param_index]
        for value in params_dict[param_name]:
            current_combination[param_name] = value
            yield from recursive_combinations(param_index + 1, current_combination)
    
    return recursive_combinations(0, {}), ordered_params


def rate_per_hour_sim_with_melting_single_threaded(
    temp, cov_mult, n_sims=100, length=256, tile_conc=TILE_CONC, tile_remaining=1.0,
    sys_fun=twelve_helix_system, kblockparams=None, safe_growth_temp=46.0,
    start_size=128*12, max_sim_time=10*3600
):
    """Single-threaded version of rate_per_hour_sim_with_melting for parallel parameter iteration"""
    from tileblockers.twelve_helix_tube import new_state
    import numpy as np
    
    if kblockparams is None:
        kblockparams = {}
    sys = sys_fun(safe_growth_temp, 0.0, 1e-7, 1.0, **kblockparams)

    max_tiles = 12 * (length - 24)  # to be safe
    ex_state = new_state(sys, length)
    min_tiles = ex_state.ntiles + 24

    states = [new_state(sys, length) for _ in range(n_sims)]
    sys.evolve(states, size_max=start_size, for_time=max_sim_time, parallel=False)

    times = np.array([state.time for state in states])
    ntiles = np.array([state.ntiles for state in states])

    sys = sys_fun(temp, cov_mult, tile_conc, tile_remaining, **kblockparams)
    for x in states:
        sys.update_state(x)
    
    sys.evolve(states, size_max=max_tiles, size_min=min_tiles, for_time=max_sim_time, parallel=False)

    times_after = np.array([state.time for state in states])
    ntiles_after = np.array([state.ntiles for state in states])

    return ((ntiles_after - ntiles) / (times_after - times)).mean() * 3600


def run_single_simulation(temp, tile_conc, bconc, n_sims=12, var_per_mean2=0.01, 
                         max_sim_time=36000, start_size=1536, length=256, sys_fun_name='simple_twelve_helix_system'):
    """Run both growth and nucleation simulation for a single parameter set"""
    import time
    
    blocker_mult = bconc / tile_conc
    
    # Map system function name to actual function
    sys_fun_map = {
        'simple_twelve_helix_system': simple_twelve_helix_system,
        'twelve_helix_system': twelve_helix_system,
        'k9_system': k9_system,
        'k10_system': k10_system
    }
    sys_fun = sys_fun_map[sys_fun_name]
    
    # Growth rate simulation with timing (using single-threaded version)
    growth_start_time = time.time()
    growth_rate = rate_per_hour_sim_with_melting_single_threaded(
        temp, blocker_mult, n_sims=n_sims, 
        length=length,
        sys_fun=sys_fun, 
        tile_conc=tile_conc, tile_remaining=1.0,
        start_size=start_size,
        max_sim_time=max_sim_time
    ) / 3600.0 / 3.5
    growth_duration = time.time() - growth_start_time
    
    # Skip nucleation simulation if growth rate is negative (real nucleation rate would be zero)
    if growth_rate < 0:
        nucleation_rate_info = (0.0, 0.0, 0.0)  # (nucleation_rate, nucleation_rate_05, nucleation_rate_95)
        nucleation_duration = 0.0
    else:
        # Nucleation rate simulation with timing
        nucleation_start_time = time.time()
        nucleation_rate_info = run_ffs_for_system(
            temp=temp, cov_mult=blocker_mult, tile_conc=tile_conc,
            var_per_mean2=var_per_mean2, min_nuc_rate=1e-14,
            sys_fun=sys_fun
        )
        nucleation_duration = time.time() - nucleation_start_time
    
    return {
        'temperature': temp,
        'tile_conc': tile_conc,
        'blocker_conc': bconc,
        'blocker_mult': blocker_mult,
        'growth_rate': growth_rate,
        'nucleation_rate': nucleation_rate_info[0],
        'nucleation_rate_05': nucleation_rate_info[1],
        'nucleation_rate_95': nucleation_rate_info[2],
        # Timing information (not included in CSV, used for progress display)
        '_growth_duration': growth_duration,
        '_nucleation_duration': nucleation_duration,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate phase diagram data with customizable parameter ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter format examples:
  Single value: --temps 45.0
  Linear range: --temps 30:55:0.5 (start:stop:step)
  Log range: --tile_concs log:1:3:20 (log:start_exp:stop_exp:num_points)
  List of values: --temps 30,45,60 (comma-separated, processed in order)

Loop ordering:
  Parameters are nested in the order specified on the command line.
  First specified parameter becomes the outer loop, last becomes inner.
  Use parameter flags without values to use defaults but control ordering.
        """
    )
    
    parser.add_argument('--temps', type=str, nargs='?', const='30:55:0.5', default='30:55:0.5',
                       help='Temperature range (default: 30:55:0.5)')
    parser.add_argument('--tile_concs', type=str, nargs='?', const='log:1:3:20', default='log:1:3:20',
                       help='Tile concentration range (default: log:1:3:20, results multiplied by 1e-9)')
    parser.add_argument('--bconcs', type=str, nargs='?', const='2.5e-6', default=None,
                       help='Blocker concentration range (default: 2.5e-6)')
    parser.add_argument('--bmults', type=str, nargs='?', const='2.5', default=None,
                       help='Blocker concentration multiplier range (ratio of blocker_conc/tile_conc)')
    parser.add_argument('--n_sims', type=int, default=12,
                       help='Number of simulations per parameter set (default: 12)')
    parser.add_argument('--var_per_mean2', type=float, default=0.01,
                       help='Variance per mean squared for nucleation rate calculations (default: 0.01)')
    parser.add_argument('--max_sim_time', type=float, default=36000,
                       help='Maximum simulation time in seconds (default: 36000, i.e., 10 hours)')
    parser.add_argument('--start_size', type=int, default=1536,
                       help='Starting size for growth simulations (default: 1536, i.e., 128*12)')
    parser.add_argument('--length', type=int, default=256,
                       help='Length parameter for simulation setup (default: 256)')
    parser.add_argument('--sys_fun', type=str, default='simple_twelve_helix_system',
                       choices=['simple_twelve_helix_system', 'twelve_helix_system', 'k9_system', 'k10_system'],
                       help='System function to use for simulations (default: simple_twelve_helix_system)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--n_threads', type=int, default=None,
                       help='Number of parallel threads (default: number of CPU cores)')
    
    # Track order of parameter specification
    import sys
    specified_params = []
    param_names = ['--temps', '--tile_concs', '--bconcs', '--bmults']
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg in param_names:
            specified_params.append(arg[2:])  # Remove --
    
    args = parser.parse_args()
    
    # Validate that exactly one of bconcs or bmults is specified
    if args.bconcs is not None and args.bmults is not None:
        parser.error("Cannot specify both --bconcs and --bmults. Use one or the other.")
    if args.bconcs is None and args.bmults is None:
        # Default to bconcs for backward compatibility
        args.bconcs = '2.5e-6'
    
    # Parse parameter ranges
    temps = parse_parameter(args.temps)
    tile_concs = np.array(parse_parameter(args.tile_concs)) * 1e-9  # Convert to M
    
    bconcs = None
    bmults = None
    if args.bconcs is not None:
        bconcs = parse_parameter(args.bconcs)
        print(f"Temperature range: {len(temps)} values from {temps[0]:.1f} to {temps[-1]:.1f}")
        print(f"Tile concentration range: {len(tile_concs)} values from {tile_concs[0]:.2e} to {tile_concs[-1]:.2e} M")
        print(f"Blocker concentration range: {len(bconcs)} values from {bconcs[0]:.2e} to {bconcs[-1]:.2e} M")
    else:
        bmults = parse_parameter(args.bmults)
        print(f"Temperature range: {len(temps)} values from {temps[0]:.1f} to {temps[-1]:.1f}")
        print(f"Tile concentration range: {len(tile_concs)} values from {tile_concs[0]:.2e} to {tile_concs[-1]:.2e} M")
        print(f"Blocker multiplier range: {len(bmults)} values from {bmults[0]:.2f} to {bmults[-1]:.2f}")
    
    # Generate output filename and path
    filename = generate_filename(temps, tile_concs, bconcs, bmults)
    output_path = Path(args.output_dir) / filename
    json_path = output_path.with_suffix('.json')
    
    print(f"Output CSV file: {output_path}")
    print(f"Output JSON file: {json_path}")
    blocker_count = len(bmults) if bmults is not None else len(bconcs)
    print(f"Total simulations: {len(temps) * len(tile_concs) * blocker_count}")
    
    # Prepare parameters for dynamic loop generation
    params_dict = {
        'temps': temps,
        'tile_concs': tile_concs,
    }
    if bconcs is not None:
        params_dict['bconcs'] = bconcs
    else:
        params_dict['bmults'] = bmults
    
    # Generate parameter combinations with specified ordering
    combinations_generator, loop_order = generate_parameter_combinations(params_dict, specified_params)
    
    # Create and save parameter information JSON
    param_info = create_parameter_info(temps, tile_concs, bconcs, bmults, args.n_sims, args.var_per_mean2, args, loop_order)
    with open(json_path, 'w') as json_file:
        json.dump(param_info, json_file, indent=2, ensure_ascii=False)
    
    # Determine number of threads
    n_threads = args.n_threads if args.n_threads is not None else os.cpu_count()
    print(f"Parameter information saved to: {json_path}")
    print(f"Parameter loop order: {' -> '.join(loop_order)}")
    print(f"Using {n_threads} parallel threads")
    
    # Prepare parameter combinations list
    combinations_list = list(combinations_generator)
    total_sims = len(combinations_list)
    
    fieldnames = ['temperature', 'tile_conc', 'blocker_conc', 'blocker_mult', 'growth_rate', 'nucleation_rate', 'nucleation_rate_05', 'nucleation_rate_95']

    def run_simulation(combination):
        """Run simulation for a single parameter combination"""
        temp = combination['temps']
        tile_conc = combination['tile_concs']

        # Handle both bconcs and bmults
        if 'bconcs' in combination:
            bconc = combination['bconcs']
        else:
            bmult = combination['bmults']
            bconc = bmult * tile_conc

        try:
            return run_single_simulation(
                temp, tile_conc, bconc, args.n_sims, args.var_per_mean2,
                args.max_sim_time, args.start_size, args.length, args.sys_fun
            )
        except Exception as e:
            print(f"\nError in simulation T={temp}, tile_conc={tile_conc:.2e}, bconc={bconc:.2e}: {e}")
            return {
                'temperature': temp,
                'tile_conc': tile_conc,
                'blocker_conc': bconc,
                'blocker_mult': bconc / tile_conc,
                'growth_rate': float('nan'),
                'nucleation_rate': float('nan'),
                'nucleation_rate_05': float('nan'),
                'nucleation_rate_95': float('nan')
            }

    # Run parallel simulations, collecting results in parameter order
    results = [None] * total_sims

    with tqdm(total=total_sims, desc="Simulations") as pbar:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_index = {
                executor.submit(run_simulation, combo): i
                for i, combo in enumerate(combinations_list)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                    if result:
                        pbar.set_postfix({
                            'T': f'{result["temperature"]:.1f}',
                            'tc': f'{result["tile_conc"]:.1e}M',
                            'bc': f'{result["blocker_conc"]:.1e}M',
                            'gr_t': f'{result.get("_growth_duration", 0):.1f}s',
                            'nr_t': f'{result.get("_nucleation_duration", 0):.1f}s',
                            'gr': f'{result["growth_rate"]:.1e}',
                            'nr': f'{result["nucleation_rate"]:.1e}'
                        })
                except Exception as e:
                    print(f"\nUnexpected error in thread: {e}")
                finally:
                    pbar.update(1)

    # Write CSV in parameter order
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if result is not None:
                csv_result = {k: v for k, v in result.items() if not k.startswith('_')}
                writer.writerow(csv_result)
    
    print(f"\nSimulations completed! Results saved to: {output_path}")
    
    # Verify the output file
    df = pl.read_csv(output_path)
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {df.columns}")


if __name__ == "__main__":
    main()