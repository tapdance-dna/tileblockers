# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses `uv` for Python package management:

- **Install dependencies**: `uv sync` (includes dev dependencies)
- **Run with virtual environment**: `uv run python <script.py>`
- **Activate virtual environment**: `source .venv/bin/activate` (then use `python` directly)
- **Run phase diagram data generation**: `uv run tileblockers-gen-data --help`
- **Run tests**: `uv run pytest`

## Architecture Overview

This is a Python scientific computing library for DNA tile blocker research, focusing on phase diagram calculations and twelve-helix tube simulations.

### Core Modules

- **`theoretical_calculations.py`**: Core thermodynamic calculations including:
  - Growth and nucleation rate calculations (`growth_rate`, `nuc_rate_rect`)  
  - Binding probability calculations (`pa_full`, `pa_approx`, `pa_full_bconc`)
  - Thermodynamic utilities (`thermo_beta`, `rt_val`, `calc_gval`)

- **`phase_diagram.py`**: Phase diagram visualization and data processing:
  - Uses Polars DataFrames for efficient data handling
  - Matplotlib-based plotting with custom arrow drawing
  - Integration with theoretical calculations via `theory_calcs()` function

- **`twelve_helix_tube.py`**: DNA tile system simulations:
  - Contains predefined tile glue sequences (`TILE_GLUE_SEQUENCES_K10`)
  - Interfaces with `rgrow` library for growth simulations
  - Rate calculations for complex tile assemblies

- **`constants.py`**: Physical constants and configuration:
  - Thermodynamic constants (`R_CONST`, `DS_LAT`)
  - Default concentrations (`TILE_CONC`)
  - DNA sequences (`SINGLE_SEQ`)
  - Color palette for visualizations (`COLORSET`)

- **`bdiviter.py`**: Binary division iterator utility for parameter space exploration

### Key Dependencies

- **Primary**: `numpy`, `matplotlib`, `polars`, `scipy`
- **Domain-specific**: `rgrow` (DNA growth simulations) - installed from GitHub
- **Development**: `ipykernel`, `ipython` for Jupyter notebook support

### Key Scripts

- **`gen_data.py`**: Phase diagram data generation script (available as `tileblockers-gen-data` command):
  - Supports multiple parameter formats: single values, ranges, logspace, comma-separated lists
  - Command-line parameter parsing for temperature/concentration ranges  
  - Batch processing of phase diagram data
  - Integration between theoretical calculations and simulation results
  - CSV output for further analysis
  - Example: `uv run tileblockers-gen-data --temps 40,50,60 --tile_concs log:1:3:5 --bconcs 0,1e-6,2e-6`

## Important Notes

- Always use virtual environment (`.venv` directory is gitignored)
- Python 3.13 is the target version (see `.python-version`)
- The codebase integrates theoretical calculations with numerical simulations
- Data processing uses Polars for performance with large datasets
- Visualization uses matplotlib with custom styling (`examples/default.mplstyle`)