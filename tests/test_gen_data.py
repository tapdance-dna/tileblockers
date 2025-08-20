#!/usr/bin/env python3
"""
Tests for the gen_data script.
Uses minimal parameter values for fast execution.
"""

import pytest
import tempfile
import polars as pl
from pathlib import Path
import subprocess
import sys
import os
import json

from tileblockers.gen_data import parse_parameter, run_single_simulation, generate_filename


class TestParseParameter:
    """Test the parameter parsing function."""
    
    def test_single_value(self):
        result = parse_parameter("2.5e-6")
        assert len(result) == 1
        assert result[0] == pytest.approx(2.5e-6)
    
    def test_range(self):
        result = parse_parameter("30:35:2.5")
        expected = [30.0, 32.5]
        assert len(result) == len(expected)
        for actual, exp in zip(result, expected):
            assert actual == pytest.approx(exp)
    
    def test_logspace(self):
        result = parse_parameter("log:1:2:3")
        expected = [10, 31.622776601683795, 100]  # 10^1, 10^1.5, 10^2
        assert len(result) == 3
        for actual, exp in zip(result, expected):
            assert actual == pytest.approx(exp, rel=1e-6)
    
    def test_list_values(self):
        result = parse_parameter("30,45,60")
        expected = [30.0, 45.0, 60.0]
        assert len(result) == len(expected)
        for actual, exp in zip(result, expected):
            assert actual == pytest.approx(exp)
    
    def test_list_values_with_spaces(self):
        result = parse_parameter("30, 45 , 60")
        expected = [30.0, 45.0, 60.0]
        assert len(result) == len(expected)
        for actual, exp in zip(result, expected):
            assert actual == pytest.approx(exp)
    
    def test_none_input(self):
        result = parse_parameter(None)
        assert result is None


class TestGenerateFilename:
    """Test filename generation."""
    
    def test_single_values(self):
        temps = [45.0]
        tile_concs = [1e-7]
        bconcs = [2.5e-6]
        filename = generate_filename(temps, tile_concs, bconcs)
        assert "T_4.50e+01" in filename
        assert "tile_1.00e-07" in filename
        assert "bconc_2.50e-06" in filename
    
    def test_ranges(self):
        temps = [30.0, 45.0, 60.0]
        tile_concs = [1e-8, 1e-7]
        bconcs = [1e-6]
        filename = generate_filename(temps, tile_concs, bconcs)
        assert "T_3.00e+01_to_6.00e+01_n3" in filename
        assert "tile_1.00e-08_to_1.00e-07_n2" in filename
        assert "bconc_1.00e-06" in filename


class TestSimulationLogic:
    """Test the core simulation functions with minimal parameters."""
    
    def test_run_single_simulation_fast(self):
        """Test simulation with minimal parameters for speed."""
        # Use parameters that should give predictable results
        temp = 45.0  # Should give positive growth rate
        tile_conc = 1e-7
        bconc = 0.0  # No blocker
        
        result = run_single_simulation(temp, tile_conc, bconc, n_sims=1, var_per_mean2=0.01)
        
        # Check that all expected keys are present
        expected_keys = ['temperature', 'tile_conc', 'blocker_conc', 'blocker_mult', 
                        'growth_rate', 'nucleation_rate', 'nucleation_rate_05', 'nucleation_rate_95']
        for key in expected_keys:
            assert key in result
        
        # Check values are reasonable
        assert result['temperature'] == temp
        assert result['tile_conc'] == tile_conc
        assert result['blocker_conc'] == bconc
        assert result['blocker_mult'] == 0.0
        assert isinstance(result['growth_rate'], (int, float))
        assert isinstance(result['nucleation_rate'], (int, float))
    
    def test_growth_rate_temperature_dependence(self):
        """Test that growth rate behaves as expected with temperature."""
        tile_conc = 1e-7
        bconc = 0.0  # No blocker for cleaner test
        
        # Test low temperature (should be positive growth rate)
        result_low = run_single_simulation(40.0, tile_conc, bconc, n_sims=1, var_per_mean2=0.01)
        
        # Test high temperature (should be negative growth rate)  
        result_high = run_single_simulation(60.0, tile_conc, bconc, n_sims=1, var_per_mean2=0.01)
        
        # At low temperature, growth should be positive (assembly favorable)
        assert result_low['growth_rate'] > 0, f"Expected positive growth at 40°C, got {result_low['growth_rate']}"
        
        # At high temperature, growth should be negative (disassembly favorable)
        assert result_high['growth_rate'] < 0, f"Expected negative growth at 60°C, got {result_high['growth_rate']}"
    
    def test_specific_growth_rate_conditions(self):
        """
        Test specific growth rate behavior as requested:
        - For tile concentration 1e-7 and blocker concentration 0:
        - Growth rate should be negative when temperature is above 55°C
        - Growth rate should be positive when temperature is below 47°C
        """
        tile_conc = 1e-7
        bconc = 0.0
        
        # Test above 55°C - should be negative
        result_high = run_single_simulation(56.0, tile_conc, bconc, n_sims=1, var_per_mean2=0.01)
        assert result_high['growth_rate'] < 0, (
            f"Expected negative growth rate at 56°C with tile_conc={tile_conc}, bconc={bconc}, "
            f"got {result_high['growth_rate']}"
        )
        
        # Test below 47°C - should be positive  
        result_low = run_single_simulation(46.0, tile_conc, bconc, n_sims=1, var_per_mean2=0.01)
        assert result_low['growth_rate'] > 0, (
            f"Expected positive growth rate at 46°C with tile_conc={tile_conc}, bconc={bconc}, "
            f"got {result_low['growth_rate']}"
        )
    
    def test_var_per_mean2_parameter(self):
        """Test that var_per_mean2 parameter is properly passed through."""
        tile_conc = 1e-7
        bconc = 0.0
        temp = 45.0
        
        # Test with different var_per_mean2 values
        result1 = run_single_simulation(temp, tile_conc, bconc, n_sims=1, var_per_mean2=0.01)
        result2 = run_single_simulation(temp, tile_conc, bconc, n_sims=1, var_per_mean2=0.05)
        
        # Both should complete successfully 
        assert isinstance(result1['nucleation_rate'], (int, float))
        assert isinstance(result2['nucleation_rate'], (int, float))
        
        # The results may be different due to the var_per_mean2 parameter
        # (though for such a simple test case they might be the same)


class TestScriptIntegration:
    """Test the complete script execution."""
    
    def test_script_runs_successfully(self):
        """Test that the script runs without errors with minimal parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run script with very minimal parameters for speed
            cmd = [
                sys.executable, "-m", "tileblockers.gen_data",
                "--temps", "45",  # Single temperature
                "--tile_concs", "1e-1",  # Single concentration (will be *1e-9 = 1e-10)
                "--bconcs", "0",  # No blocker
                "--n_sims", "1",  # Minimal simulations
                "--output_dir", tmpdir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/var/home/const/repos/tileblockers")
            
            # Check that script completed successfully
            assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
            
            # Check that output files were created
            csv_files = list(Path(tmpdir).glob("*.csv"))
            json_files = list(Path(tmpdir).glob("*.json"))
            assert len(csv_files) == 1, f"Expected 1 CSV file, found {len(csv_files)}"
            assert len(json_files) == 1, f"Expected 1 JSON file, found {len(json_files)}"
            
            # Check that CSV file has expected structure
            df = pl.read_csv(csv_files[0])
            expected_columns = ['temperature', 'tile_conc', 'blocker_conc', 'blocker_mult', 
                              'growth_rate', 'nucleation_rate', 'nucleation_rate_05', 'nucleation_rate_95']
            for col in expected_columns:
                assert col in df.columns, f"Missing column: {col}"
            
            assert len(df) == 1, f"Expected 1 row of data, got {len(df)}"
            
            # Check that JSON file has expected structure
            with open(json_files[0]) as f:
                json_data = json.load(f)
            
            expected_json_keys = ["generation_info", "simulation_parameters", "parameter_ranges", "output_info"]
            for key in expected_json_keys:
                assert key in json_data, f"Missing JSON key: {key}"
            
            assert json_data["simulation_parameters"]["n_sims_per_point"] == 1
            assert json_data["simulation_parameters"]["var_per_mean2"] == 0.01
    
    def test_script_with_list_parameters(self):
        """Test that the script handles list parameters correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable, "-m", "tileblockers.gen_data",
                "--temps", "40,50",  # Two temperatures
                "--tile_concs", "0.1,1",  # Two concentrations  
                "--bconcs", "0",  # No blocker
                "--n_sims", "1",
                "--output_dir", tmpdir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/var/home/const/repos/tileblockers")
            
            assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
            
            # Check output file
            output_files = list(Path(tmpdir).glob("*.csv"))
            assert len(output_files) == 1
            
            df = pl.read_csv(output_files[0])
            assert len(df) == 4, f"Expected 4 rows (2 temps × 2 tile_concs), got {len(df)}"
            
            # Check that temperatures are in the expected order
            temps = df['temperature'].to_list()
            expected_temp_order = [40.0, 40.0, 50.0, 50.0]  # nested loop order
            assert temps == expected_temp_order
    
    def test_script_with_var_per_mean2_parameter(self):
        """Test that the script accepts and uses the --var_per_mean2 parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable, "-m", "tileblockers.gen_data",
                "--temps", "45",
                "--tile_concs", "0.1",
                "--bconcs", "0",
                "--n_sims", "1",
                "--var_per_mean2", "0.05",  # Non-default value
                "--output_dir", tmpdir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/var/home/const/repos/tileblockers")
            
            assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
            
            # Check output files were created successfully
            csv_files = list(Path(tmpdir).glob("*.csv"))
            json_files = list(Path(tmpdir).glob("*.json"))
            assert len(csv_files) == 1
            assert len(json_files) == 1
            
            df = pl.read_csv(csv_files[0])
            assert len(df) == 1
            
            # Verify the nucleation rate is present (indicates simulation ran)
            assert 'nucleation_rate' in df.columns
            assert not df['nucleation_rate'][0] != df['nucleation_rate'][0]  # Not NaN
            
            # Check that JSON contains the correct var_per_mean2 value
            with open(json_files[0]) as f:
                json_data = json.load(f)
            assert json_data["simulation_parameters"]["var_per_mean2"] == 0.05
    
    def test_parameter_loop_ordering(self):
        """Test that parameter loop ordering works correctly based on command-line specification order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Specify parameters in non-default order: bconcs, temps, tile_concs
            cmd = [
                sys.executable, "-m", "tileblockers.gen_data",
                "--bconcs", "0,1e-6",  # Two blocker concentrations (outer loop)
                "--temps", "40,50",    # Two temperatures (middle loop) 
                "--tile_concs", "0.1,1",  # Two tile concentrations (inner loop)
                "--n_sims", "1",
                "--output_dir", tmpdir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/var/home/const/repos/tileblockers")
            
            assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
            
            # Check output files
            csv_files = list(Path(tmpdir).glob("*.csv"))
            json_files = list(Path(tmpdir).glob("*.json"))
            assert len(csv_files) == 1
            assert len(json_files) == 1
            
            df = pl.read_csv(csv_files[0])
            assert len(df) == 8, f"Expected 8 rows (2 bconcs × 2 temps × 2 tile_concs), got {len(df)}"
            
            # Check that the loop order is preserved: bconcs (outer) -> temps -> tile_concs (inner)
            # First 4 rows should have bconc=0, next 4 should have bconc=1e-6
            bconcs = df['blocker_conc'].to_list()
            expected_bconc_pattern = [0.0] * 4 + [1e-6] * 4
            assert bconcs == expected_bconc_pattern, f"Expected {expected_bconc_pattern}, got {bconcs}"
            
            # Within each bconc group, temps should be the next loop level
            # First 2 rows: bconc=0, temp=40; next 2: bconc=0, temp=50; etc.
            temps = df['temperature'].to_list()
            expected_temp_pattern = [40.0, 40.0, 50.0, 50.0, 40.0, 40.0, 50.0, 50.0]
            assert temps == expected_temp_pattern, f"Expected {expected_temp_pattern}, got {temps}"
            
            # Check JSON has the correct loop order
            with open(json_files[0]) as f:
                json_data = json.load(f)
            
            expected_loop_order = ['bconcs', 'temps', 'tile_concs']
            assert json_data["generation_info"]["loop_order"] == expected_loop_order
    
    def test_default_parameter_flags_without_values(self):
        """Test using parameter flags without values to control ordering while using defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use --bconcs first (no value = use default), then --temps (with value)
            cmd = [
                sys.executable, "-m", "tileblockers.gen_data",
                "--bconcs",            # No value, uses default 2.5e-6
                "--temps", "40,45",    # Specified value
                "--tile_concs", "0.1", # Single value for speed
                "--n_sims", "1",
                "--output_dir", tmpdir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/var/home/const/repos/tileblockers")
            
            assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
            
            # Check that JSON shows the correct loop order
            json_files = list(Path(tmpdir).glob("*.json"))
            assert len(json_files) == 1
            
            with open(json_files[0]) as f:
                json_data = json.load(f)
            
            # Loop order should be bconcs (first specified), temps (second), tile_concs (third)
            expected_loop_order = ['bconcs', 'temps', 'tile_concs']
            assert json_data["generation_info"]["loop_order"] == expected_loop_order


if __name__ == "__main__":
    pytest.main([__file__])