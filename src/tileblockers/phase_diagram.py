import polars as pl
import sys
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .theoretical_calculations import (
    pa_full,
    pa_full_bconc,
    pa_approx,
    growth_rate,
    calc_gval,
    thermo_beta,
    rt_val,
    nuc_rate_rect,
)

def draw_arrows(ax, coords, zorder=7, **arrowprops):
    merged = dict(arrowstyle="->", color="black", lw=1, shrinkA=0, shrinkB=0) | arrowprops
    for (start, end) in zip(coords, coords[1:]):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=merged,
            zorder=zorder,
        )
    ax.plot(*coords[0], "o", color=merged["color"], markersize=3, zorder=zorder)


def theory_calcs(df, adj_bdg37: float = 0.0, adj_bds: float = 0.0):
    return df.with_columns(
        pl.struct("temperature", "tile_conc", "blocker_conc")
        .map_elements(
            lambda x: growth_rate(
                x["temperature"], x["blocker_conc"] / x["tile_conc"], x["tile_conc"], adj_bdg=adj_bdg37, adj_bds=adj_bds
            ),
            return_dtype=pl.Float64,
        )
        .alias("growth_rate"),
        pl.struct("temperature", "tile_conc", "blocker_conc")
        .map_elements(
            lambda x: nuc_rate_rect(
                x["temperature"], x["blocker_conc"] / x["tile_conc"], x["tile_conc"], adj_bdg=adj_bdg37, adj_bds=adj_bds
            ),
            return_dtype=pl.Float64,
        )
        .alias("nucleation_rate"),
        pl.struct("temperature", "tile_conc", "blocker_conc")
        .map_elements(
            lambda x: growth_rate(
                x["temperature"],
                x["blocker_conc"] / x["tile_conc"],
                x["tile_conc"],
                bonds=1,
                adj_bdg=adj_bdg37,
                adj_bds=adj_bds,
            ),
            return_dtype=pl.Float64,
        )
        .alias("growth_rate_1bond"),
        pl.struct("temperature", "tile_conc", "blocker_conc")
        .map_elements(
            lambda x: pa_full_bconc(
                x["temperature"],
                x["blocker_conc"],
                x["tile_conc"],
                adj_bdg=adj_bdg37,
                adj_bds=adj_bds,
            ),
            return_dtype=pl.Float64,
        )
        .alias("pa"),
    )

def value_df(temps, tile_concs, blocker_concs=None, blocker_mults=None):
    df = pl.DataFrame({"temperature": temps})
    df = df.join(pl.DataFrame({"tile_conc": tile_concs}), how="cross")
    match blocker_concs, blocker_mults:
        case (x, None):
            df = df.join(pl.DataFrame({"blocker_conc": x}), how="cross")
            df = df.with_columns(
                (pl.col("blocker_conc") * pl.col("tile_conc")).alias("blocker_mult")
            )
        case (None, x):
            df = df.join(pl.DataFrame({"blocker_mult": x}), how="cross")
            df = df.with_columns(
                (pl.col("blocker_mult") * pl.col("tile_conc")).alias("blocker_conc")
            )
        case _:
            raise ValueError(
                "Exactly one of blocker_concs or blocker_mults must be provided"
            )
    return df

def draw_phase_diagram(
    df,
    x_val,
    y_val,
    filt=None,
    ax=None,
    growth_type: Literal["contour", "heatmap"] = "contour",
    nuc_type: Literal["contour", "heatmap"] = "contour",
    include_melting: bool = True,
    include_growth_rates: bool | list[float] = True,
    include_growth1_rates: bool | list[float] = True,
    include_nucleation: bool | list[float] = True,
    agg="mean"
):
    if filt is not None:
        vals = df.filter(filt)
    else:
        vals = df

    # Filter out rows with null growth_rate values
    vals = vals.filter(pl.col("growth_rate").is_not_null())

    if len(vals) == 0:
        raise ValueError("No valid data remaining after filtering out null values")

    # Check for complete grid
    unique_x = vals[x_val].unique()
    unique_y = vals[y_val].unique()
    expected_combos = len(unique_x) * len(unique_y)
    actual_combos = vals.select(x_val, y_val).unique().height
    if actual_combos != expected_combos:
        raise ValueError("Incomplete data grid after null filtering")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Create pivot tables for coordinate grids - handle potential NaN/missing values
    try:
        xg = (
            vals.pivot(index=x_val, on=y_val, values=x_val, aggregate_function=agg)
            .select(pl.exclude(x_val))
            .to_numpy()
        )
        yg = (
            vals.pivot(index=x_val, on=y_val, values=y_val, aggregate_function=agg)
            .select(pl.exclude(x_val))
            .to_numpy()
        )

        gv = (
            vals.pivot(index=x_val, on=y_val, values="growth_rate", aggregate_function=agg)
            .select(pl.exclude(x_val))
            .to_numpy()
        )
        

    except Exception as e:
        raise ValueError(f"Failed to create pivot tables for plotting. This may be due to incomplete data grid after null filtering: {e}")


    growthrates_h = gv * 3600

    growth_contour_levels = []
    growth_contour_colors = []

    if include_melting:
        growth_contour_levels.append(-np.inf)
        growth_contour_levels += [-100, -10**1.5, -10, -10**0.5, -1] # , 0
        growth_contour_colors += [
            "#e04c3c",  # most intense red
            "#f06d5c",
            "#f28e7c",
            "#f4af9c",
            "#f6d0bc",
            # "#9d9494"   # least intense, grayish
        ]

    growth_available_colors=[
                        "#949d94",
                        "#b2d8b2",
                        "#7fbf7f",
                        "#4caf50",
                        "#009000",
                        "#005000",
                    ]
    avail_iter = iter(growth_available_colors)

    if include_growth_rates:
        if isinstance(include_growth_rates, bool):
            include_growth_rates = [1, 10**0.5, 10, 10**1.5, 100, np.inf]
        for rate in include_growth_rates:
            growth_contour_levels.append(rate)
            growth_contour_colors.append(next(avail_iter))

    match growth_type:
        case "contour":
            ax.contourf(
                xg,
                yg,
                growthrates_h,
                colors=growth_contour_colors,
                levels=growth_contour_levels,
                zorder=1,
            )
            # Thin contour lines to visually distinguish fills
            finite_growth_levels = [l for l in growth_contour_levels if np.isfinite(l)]
            ax.contour(xg, yg, growthrates_h, levels=finite_growth_levels, colors='black', linewidths=0.2, linestyles='solid', zorder=2)
        case "heatmap":
            from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

            colors = ["#99a099", "#b2d8b2", "#7fbf7f", "#4caf50", "#006400"]
            cmap = LinearSegmentedColormap.from_list("custom_contour", colors, N=256)
            masked = np.ma.masked_where(growthrates_h <= 0, growthrates_h)
            ax.pcolormesh(xg, yg, masked, cmap=cmap, vmax=100)

            ax.contourf(
                xg, yg, growthrates_h, colors=["#e04c3c"], levels=[-1e30, -1e-30]
            )
            ax.contourf(xg, yg, growthrates_h, colors=["#99a099"], levels=[-1e-30, 1])
    # ax.contour(xg, yg, growthrates_h, levels=[-1, 1], colors="black", linewidths=1)


    if include_nucleation:

        spont_nuc = (
            vals.pivot(index=x_val, on=y_val, values="nucleation_rate", aggregate_function=agg)
            .select(pl.exclude(x_val))
            .to_numpy()
        )
        

        avail_nuc_colors = [
            # "#ede7f680",  # lightest, most transparent
            "#b39ddbff", #cc",
            "#7e57c2ff", #e6",
            "#5e35b1ff", #f0",
            "#4527a0ff", #fa",
            "#311b92ff", #ff"  # most saturated, fully opaque
        ]
        avail_nuc_iter = iter(avail_nuc_colors)
        spont_nuc_contour_levels = []
        spont_nuc_contour_colors = []
        if isinstance(include_nucleation, bool):
            include_nucleation = [1e-12, 1e-9, 1e-6, 1e-3, np.inf]
        for rate in include_nucleation:
            spont_nuc_contour_levels.append(rate)
            spont_nuc_contour_colors.append(next(avail_nuc_iter))
        match nuc_type:
            case "contour":
                ax.contourf(xg, yg, spont_nuc, colors=spont_nuc_contour_colors, levels=spont_nuc_contour_levels, zorder=3)
                # Thin contour lines to visually distinguish fills
                finite_nuc_levels = [l for l in spont_nuc_contour_levels if np.isfinite(l)]
                ax.contour(xg, yg, spont_nuc, levels=finite_nuc_levels, colors='black', linewidths=0.2, linestyles='solid', zorder=4)
            case "heatmap":
                from matplotlib.colors import LinearSegmentedColormap

                colors = ["#f5d6a7", "#ff9800"]
                cmap = LinearSegmentedColormap.from_list("custom_contour", colors, N=256)
                masked = np.ma.masked_where(spont_nuc <= 1e-6, spont_nuc)
                ax.pcolormesh(xg, yg, masked, cmap=cmap, vmin=1e-6, vmax=1e-4)

        # ax.contour(xg, yg, spont_nuc, levels=[1e-6], colors="black", linewidths=1)


    if include_growth1_rates:
        if "growth_rate_1bond" not in vals.columns:
            import warnings
            warnings.warn("growth_rate_1bond column missing, skipping 1-bond growth rate overlay")
        else:
            avail1_colors = ["#ffe0b2", "#ffe082", "#ffd54f", "#ffca28", "#ffc107", "#ffb300"]
            avail1_iter = iter(avail1_colors)
            if isinstance(include_growth1_rates, bool):
                include_growth1_rates = [0, 10, 50, 100, 500, np.inf]
            gv1 = (
                vals.pivot(index=x_val, on=y_val, values="growth_rate_1bond", aggregate_function=agg)
                .select(pl.exclude(x_val))
                .to_numpy()
            ) * 3600

            growth1_contour_levels = []
            growth1_contour_colors = []
            if isinstance(include_growth1_rates, bool):
                growth1_contour_levels.append(0)
                growth1_contour_colors.append("#ff9800")
            for rate in include_growth1_rates:
                growth1_contour_levels.append(rate)
                growth1_contour_colors.append(next(avail1_iter))
            ax.contourf(xg, yg, gv1, colors=growth1_contour_colors, levels=growth1_contour_levels, zorder=5)
            # Thin contour lines to visually distinguish fills
            finite_growth1_levels = [l for l in growth1_contour_levels if np.isfinite(l)]
            ax.contour(xg, yg, gv1, levels=finite_growth1_levels, colors='black', linewidths=0.2, linestyles='solid', zorder=6)

    labeldict = {
        "temperature": "Temperature (°C)",
        "tile_conc": "Tile concentration (nM)",
        "blocker_conc": "Blocker concentration (nM)",
        "blocker_mult": "Blocker conc ÷ tile conc",

    }

    ax.set_xlabel(labeldict[x_val])
    ax.set_ylabel(labeldict[y_val])

    if y_val in ("tile_conc", "blocker_conc"):
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*1e9:.0f}"))
        print("Set y axis to nM")
    if x_val in ("tile_conc", "blocker_conc"):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*1e9:.0f}"))
        print("Set x axis to nM")

    for label in ax.get_yticklabels():
        label.set_rotation(90)
        # label.set_ha("center")
        label.set_va("center")

    return ax
