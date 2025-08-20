import polars as pl
import sys
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("figures-generation/fig1")

from blockers_sim_paper_support.theoretical_calculations import (
    pa_full,
    pa_full_bconc,
    pa_approx,
    growth_rate,
    calc_gval,
    thermo_beta,
    rt_val,
    nuc_rate_rect,
)

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

def phase_diagram(
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
):
    if filt is not None:
        vals = df.filter(filt)
    else:
        vals = df

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    xg = (
        vals.pivot(index=x_val, on=y_val, values=x_val)
        .select(pl.exclude(x_val))
        .to_numpy()
    )
    yg = (
        vals.pivot(index=x_val, on=y_val, values=y_val)
        .select(pl.exclude(x_val))
        .to_numpy()
    )

    gv = (
        vals.pivot(index=x_val, on=y_val, values="growth_rate")
        .select(pl.exclude(x_val))
        .to_numpy()
    )


    growthrates_h = gv * 3600

    growth_contour_levels = []
    growth_contour_colors = []

    if include_melting:
        growth_contour_levels.append(-np.inf)
        growth_contour_levels.append(-1)
        growth_contour_colors.append("#e04c3c")
        growth_contour_levels.append(-1e-30)
        growth_contour_colors.append("#fbe9f0")  # less saturated, very pale red

    growth_available_colors=[
        "gray",
                        "#99a099",
                        "#b2d8b2",
                        "#7fbf7f",
                        "#4caf50",
                        "#006400",
                        "#000000",
                    ]
    avail_iter = iter(growth_available_colors)

    if include_growth_rates:
        if isinstance(include_growth_rates, bool):
            include_growth_rates = [0, 1, 10, 50, 100, np.inf]
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
            )
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
    ax.contour(xg, yg, growthrates_h, levels=[0], colors="black", linewidths=1)

    spont_nuc = (
        vals.pivot(index=x_val, on=y_val, values="nucleation_rate")
        .select(pl.exclude(x_val))
        .to_numpy()
    )

    avail_nuc_colors = [
        "#ede7f680",  # lightest, most transparent
        "#b39ddbcc",
        "#7e57c2e6",
        "#5e35b1f0",
        "#4527a0fa",
        "#311b92ff"  # most saturated, fully opaque
    ]
    avail_nuc_iter = iter(avail_nuc_colors)

    if include_nucleation:
        spont_nuc_contour_levels = []
        spont_nuc_contour_colors = []
        if isinstance(include_nucleation, bool):
            include_nucleation = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, np.inf]
        for rate in include_nucleation:
            spont_nuc_contour_levels.append(rate)
            spont_nuc_contour_colors.append(next(avail_nuc_iter))
        match nuc_type:
            case "contour":
                ax.contourf(xg, yg, spont_nuc, colors=spont_nuc_contour_colors, levels=spont_nuc_contour_levels)
            case "heatmap":
                from matplotlib.colors import LinearSegmentedColormap

                colors = ["#f5d6a7", "#ff9800"]
                cmap = LinearSegmentedColormap.from_list("custom_contour", colors, N=256)
                masked = np.ma.masked_where(spont_nuc <= 1e-6, spont_nuc)
                ax.pcolormesh(xg, yg, masked, cmap=cmap, vmin=1e-6, vmax=1e-4)

        # ax.contour(xg, yg, spont_nuc, levels=[1e-6], colors="black", linewidths=1)

    avail1_colors = ["#ffe0b2", "#ffb74d", "#ff9800", "#f57c00", "#e65100", "#bf360c"]
    avail1_iter = iter(avail1_colors)

    if include_growth1_rates:
        if isinstance(include_growth1_rates, bool):
            include_growth1_rates = [0, 10, 50, 100, 500, np.inf]
        gv1 = (
            vals.pivot(index=x_val, on=y_val, values="growth_rate_1bond")
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
        ax.contourf(xg, yg, gv1, colors=growth1_contour_colors, levels=growth1_contour_levels)
        ax.contour(xg, yg, gv1, levels=growth1_contour_levels, colors="black", linewidths=1)

    ax.set_xlabel(x_val)
    ax.set_ylabel(y_val)

    return ax
