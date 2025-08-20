#!/usr/bin/env python3
"""
Generate phase diagram data with growth and nucleation rates.
Supports command line arguments for parameter ranges and outputs results line-by-line.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from tileblockers.twelve_helix_tube import (
    rate_per_hour_sim_with_melting, 
    run_ffs_for_system, 
    simple_twelve_helix_system
)
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


def generate_filename(temps, tile_concs, bconcs):
    """Generate filename based on parameter ranges"""
    def format_range(values, name):
        if len(values) == 1:
            return f"{name}_{values[0]:.2e}"
        else:
            return f"{name}_{values[0]:.2e}_to_{values[-1]:.2e}_n{len(values)}"
    
    temp_str = format_range(temps, "T")
    tile_str = format_range(tile_concs, "tile")
    bconc_str = format_range(bconcs, "bconc")
    
    return f"phase_diagram_data_{temp_str}_{tile_str}_{bconc_str}.csv"


def create_parameter_info(temps, tile_concs, bconcs, n_sims, var_per_mean2, args, loop_order=None):
    """Create parameter information dictionary for JSON output"""
    return {
        "generation_info": {
            "timestamp": datetime.now().isoformat(),
            "script_version": "tileblockers-gen-data",
            "total_simulations": len(temps) * len(tile_concs) * len(bconcs),
            "loop_order": loop_order or ['temps', 'tile_concs', 'bconcs'],
            "loop_order_note": "Parameters are nested in this order (first = outer loop, last = inner loop)"
        },
        "simulation_parameters": {
            "n_sims_per_point": n_sims,
            "var_per_mean2": var_per_mean2
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
                "values": bconcs.tolist() if hasattr(bconcs, 'tolist') else list(bconcs),
                "unit": "M",
                "count": len(bconcs),
                "range": f"{bconcs[0]:.2e} to {bconcs[-1]:.2e}" if len(bconcs) > 1 else f"{bconcs[0]:.2e}",
                "original_spec": args.bconcs
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
    default_order = ['temps', 'tile_concs', 'bconcs']
    
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


def run_single_simulation(temp, tile_conc, bconc, n_sims=12, var_per_mean2=0.01):
    """Run both growth and nucleation simulation for a single parameter set"""
    blocker_mult = bconc / tile_conc
    
    # Growth rate simulation
    growth_rate = rate_per_hour_sim_with_melting(
        temp, blocker_mult, n_sims=n_sims, 
        sys_fun=simple_twelve_helix_system, 
        tile_conc=tile_conc, tile_remaining=1.0
    ) / 3600.0 / 3.5
    
    # Nucleation rate simulation
    nucleation_rate_info = run_ffs_for_system(
        temp=temp, cov_mult=blocker_mult, tile_conc=tile_conc,
        var_per_mean2=var_per_mean2, min_nuc_rate=1e-14,
        sys_fun=simple_twelve_helix_system
    )
    
    return {
        'temperature': temp,
        'tile_conc': tile_conc,
        'blocker_conc': bconc,
        'blocker_mult': blocker_mult,
        'growth_rate': growth_rate,
        'nucleation_rate': nucleation_rate_info[0],
        'nucleation_rate_05': nucleation_rate_info[1],
        'nucleation_rate_95': nucleation_rate_info[2],
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
    parser.add_argument('--bconcs', type=str, nargs='?', const='2.5e-6', default='2.5e-6',
                       help='Blocker concentration range (default: 2.5e-6)')
    parser.add_argument('--n_sims', type=int, default=12,
                       help='Number of simulations per parameter set (default: 12)')
    parser.add_argument('--var_per_mean2', type=float, default=0.01,
                       help='Variance per mean squared for nucleation rate calculations (default: 0.01)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory (default: current directory)')
    
    # Track order of parameter specification
    import sys
    specified_params = []
    param_names = ['--temps', '--tile_concs', '--bconcs']
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg in param_names:
            specified_params.append(arg[2:])  # Remove --
    
    args = parser.parse_args()
    
    # Parse parameter ranges
    temps = parse_parameter(args.temps)
    tile_concs = np.array(parse_parameter(args.tile_concs)) * 1e-9  # Convert to M
    bconcs = parse_parameter(args.bconcs)
    
    print(f"Temperature range: {len(temps)} values from {temps[0]:.1f} to {temps[-1]:.1f}")
    print(f"Tile concentration range: {len(tile_concs)} values from {tile_concs[0]:.2e} to {tile_concs[-1]:.2e} M")
    print(f"Blocker concentration range: {len(bconcs)} values from {bconcs[0]:.2e} to {bconcs[-1]:.2e} M")
    
    # Generate output filename and path
    filename = generate_filename(temps, tile_concs, bconcs)
    output_path = Path(args.output_dir) / filename
    json_path = output_path.with_suffix('.json')
    
    print(f"Output CSV file: {output_path}")
    print(f"Output JSON file: {json_path}")
    print(f"Total simulations: {len(temps) * len(tile_concs) * len(bconcs)}")
    
    # Prepare parameters for dynamic loop generation
    params_dict = {
        'temps': temps,
        'tile_concs': tile_concs,
        'bconcs': bconcs
    }
    
    # Generate parameter combinations with specified ordering
    combinations_generator, loop_order = generate_parameter_combinations(params_dict, specified_params)
    
    # Create and save parameter information JSON
    param_info = create_parameter_info(temps, tile_concs, bconcs, args.n_sims, args.var_per_mean2, args, loop_order)
    with open(json_path, 'w') as json_file:
        json.dump(param_info, json_file, indent=2, ensure_ascii=False)
    
    print(f"Parameter information saved to: {json_path}")
    print(f"Parameter loop order: {' -> '.join(loop_order)}")
    
    # Prepare CSV file and write header
    fieldnames = ['temperature', 'tile_conc', 'blocker_conc', 'blocker_mult', 'growth_rate', 'nucleation_rate', 'nucleation_rate_05', 'nucleation_rate_95']
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Run simulations and write results line by line
        total_sims = len(temps) * len(tile_concs) * len(bconcs)
        
        with tqdm(total=total_sims, desc="Simulations") as pbar:
            for combination in combinations_generator:
                temp = combination['temps']
                tile_conc = combination['tile_concs'] 
                bconc = combination['bconcs']
                
                try:
                    result = run_single_simulation(temp, tile_conc, bconc, args.n_sims, args.var_per_mean2)
                    writer.writerow(result)
                    csvfile.flush()  # Ensure data is written immediately
                    
                    pbar.set_postfix({
                        'T': f'{temp:.1f}°C',
                        'tile': f'{tile_conc:.1e}M',
                        'bconc': f'{bconc:.1e}M'
                    })
                    
                except Exception as e:
                    print(f"\nError in simulation T={temp}, tile_conc={tile_conc:.2e}, bconc={bconc:.2e}: {e}")
                    # Write a row with NaN values to maintain structure
                    result = {
                        'temperature': temp,
                        'tile_conc': tile_conc,
                        'blocker_conc': bconc,
                        'blocker_mult': bconc / tile_conc,
                        'growth_rate': float('nan'),
                        'nucleation_rate': float('nan')
                    }
                    writer.writerow(result)
                    csvfile.flush()
                    raise e
                
                pbar.update(1)
    
    print(f"\nSimulations completed! Results saved to: {output_path}")
    
    # Verify the output file
    df = pl.read_csv(output_path)
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {df.columns}")


if __name__ == "__main__":
    main()