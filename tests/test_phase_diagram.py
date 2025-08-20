#!/usr/bin/env python3
"""
Tests for the phase_diagram module.
Focuses on handling null values and incomplete data.
"""

import pytest
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from tileblockers.phase_diagram import draw_phase_diagram, value_df


class TestPhaseDigramNullHandling:
    """Test that phase diagram functions handle null values correctly."""
    
    def test_draw_phase_diagram_with_complete_data(self):
        """Test that draw_phase_diagram works with complete data."""
        # Create complete test data
        temps = [40.0, 45.0, 50.0]
        tile_concs = [1e-9, 2e-9]
        blocker_concs = [0.0, 1e-6]
        
        df = value_df(temps, tile_concs, blocker_concs=blocker_concs)
        
        # Add some synthetic rate data
        df = df.with_columns([
            pl.lit(1.0).alias("growth_rate"),
            pl.lit(1e-6).alias("nucleation_rate"),
            pl.lit(0.5).alias("growth_rate_1bond")
        ])
        
        # Should work without issues
        fig, ax = plt.subplots()
        result_ax = draw_phase_diagram(df, "temperature", "tile_conc", ax=ax)
        assert result_ax is ax
        plt.close(fig)
    
    def test_draw_phase_diagram_with_null_values(self):
        """Test that draw_phase_diagram handles null values correctly."""
        # Create test data with some null values but ensure complete grid coverage after filtering
        # We need ALL combinations of x_val and y_val to be present after null filtering for contour plots to work
        df = pl.DataFrame({
            "temperature": [40.0, 40.0, 45.0, 45.0, 40.0, 40.0, 45.0, 45.0],  # 2 temps
            "tile_conc": [1e-9, 2e-9, 1e-9, 2e-9, 1e-9, 2e-9, 1e-9, 2e-9],  # 2 concentrations  
            "blocker_conc": [0.0, 0.0, 0.0, 0.0, 1e-6, 1e-6, 1e-6, 1e-6],
            "growth_rate": [1.0, 2.0, 3.0, 4.0, None, None, None, None],  # First 4 are valid, complete 2x2 grid
            "nucleation_rate": [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6],  # All valid
            "growth_rate_1bond": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # All valid
        })
        
        # Should work by filtering out rows with null values - the first 4 rows form a complete 2x2 grid
        fig, ax = plt.subplots()
        result_ax = draw_phase_diagram(df, "temperature", "tile_conc", ax=ax)
        assert result_ax is ax
        plt.close(fig)
    
    def test_draw_phase_diagram_all_null_values(self):
        """Test that draw_phase_diagram raises appropriate error when all data is null."""
        # Create test data where all growth_rate values are null
        df = pl.DataFrame({
            "temperature": [40.0, 45.0, 50.0],
            "tile_conc": [1e-9, 2e-9, 3e-9],
            "blocker_conc": [0.0, 1e-6, 2e-6],
            "growth_rate": [None, None, None],  # All null
            "nucleation_rate": [1e-6, 2e-6, 3e-6],
            "growth_rate_1bond": [0.5, 0.6, 0.7]
        })
        
        # Should raise ValueError
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="No valid data remaining after filtering out null values"):
            draw_phase_diagram(df, "temperature", "tile_conc", ax=ax)
        plt.close(fig)
    
    def test_draw_phase_diagram_partial_null_nucleation(self):
        """Test handling when nucleation data has nulls but growth data is complete."""
        df = pl.DataFrame({
            "temperature": [40.0, 45.0, 50.0, 55.0, 40.0, 45.0, 50.0, 55.0],
            "tile_conc": [1e-9, 1e-9, 1e-9, 1e-9, 2e-9, 2e-9, 2e-9, 2e-9],
            "blocker_conc": [0.0, 0.0, 0.0, 0.0, 1e-6, 1e-6, 1e-6, 1e-6],
            "growth_rate": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # Complete
            "nucleation_rate": [1e-6, None, 2e-6, None, None, 5e-6, None, 7e-6],  # Partial nulls
            "growth_rate_1bond": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # Complete
        })
        
        # Should work - nucleation will be plotted only where data exists
        fig, ax = plt.subplots()
        result_ax = draw_phase_diagram(df, "temperature", "tile_conc", ax=ax, include_nucleation=True)
        assert result_ax is ax
        plt.close(fig)
    
    def test_draw_phase_diagram_skip_nucleation_with_nulls(self):
        """Test that nucleation nulls don't matter when nucleation plotting is disabled."""
        df = pl.DataFrame({
            "temperature": [40.0, 45.0, 50.0, 55.0, 40.0, 45.0, 50.0, 55.0],
            "tile_conc": [1e-9, 1e-9, 1e-9, 1e-9, 2e-9, 2e-9, 2e-9, 2e-9],
            "blocker_conc": [0.0, 0.0, 0.0, 0.0, 1e-6, 1e-6, 1e-6, 1e-6],
            "growth_rate": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # Complete
            "nucleation_rate": [None, None, None, None, None, None, None, None],  # All null
            "growth_rate_1bond": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # Complete
        })
        
        # Should work when nucleation plotting is disabled
        fig, ax = plt.subplots()
        result_ax = draw_phase_diagram(df, "temperature", "tile_conc", ax=ax, include_nucleation=False)
        assert result_ax is ax
        plt.close(fig)
    
    def test_incomplete_grid_data(self):
        """Test handling when the 2D parameter grid has missing combinations."""
        # Create data that has incomplete grid coverage (not all temp/tile_conc combinations present)
        df = pl.DataFrame({
            "temperature": [40.0, 40.0, 45.0],  # Missing 45°C + 2e-9 combination
            "tile_conc": [1e-9, 2e-9, 1e-9],   # Missing 45°C + 2e-9 combination
            "blocker_conc": [0.0, 0.0, 0.0],
            "growth_rate": [1.0, 2.0, 3.0],
            "nucleation_rate": [1e-6, 2e-6, 3e-6],
            "growth_rate_1bond": [0.1, 0.2, 0.3]
        })
        
        # Should raise error due to incomplete grid
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Incomplete data grid after null filtering"):
            draw_phase_diagram(df, "temperature", "tile_conc", ax=ax)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])