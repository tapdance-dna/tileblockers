import numpy as np
from rgrow.rgrow import string_dna_dg_ds
from .constants import R_CONST, TILE_CONC, SINGLE_SEQ, DS_LAT, K_F

def calc_gval(seq=None, temp=37, dg=None, dg37=None, ds=None, adj_bdg37=0.0, adj_bds=0.0):
    if dg is not None:
        return dg
    elif seq is not None:
        gsval37 = string_dna_dg_ds(seq)
        return gsval37[0] - (temp - 37) * (gsval37[1] + adj_bds) + adj_bdg37
    elif dg37 is not None and ds is not None:
        return dg37 - (temp - 37) * (ds + adj_bds) + adj_bdg37
    else:
        raise ValueError("Either seq, dg, or both dg37 and ds must be provided")

def thermo_beta(temp):
    "Calculates thermodynamic beta in 1/kcal/mol from temperature in Celsius."
    return 1 / (R_CONST * (temp + 273.15))

def rt_val(temp):
    "Calculates RT in kcal/mol from temperature in Celsius."
    return R_CONST * (temp + 273.15)

def pa_approx(temp, cov_mult, tile_conc = TILE_CONC, gseq = SINGLE_SEQ, dg=None):
    beta = thermo_beta(temp)
    gval = calc_gval(seq=gseq, temp=temp, dg=dg)
    return 1 / (1 + cov_mult * tile_conc * np.exp(-beta * gval))

def pa_full(temp, cov_mult, tile_conc = TILE_CONC, gseq = SINGLE_SEQ, dg=None, adj_bdg=0.0, adj_bds=0.0):
    beta = thermo_beta(temp)
    gval = calc_gval(seq=gseq, temp=temp, dg=dg, adj_bdg37=adj_bdg, adj_bds=adj_bds)
    b_conc = cov_mult * tile_conc
    gb = np.exp(beta * gval)
    return (0.5 * (tile_conc - b_conc - gb + np.sqrt((b_conc - tile_conc + gb)**2 + 4 * tile_conc * gb))) / tile_conc

def pa_full_bconc(temp, blocker_conc, tile_conc = TILE_CONC, gseq = SINGLE_SEQ, dg=None, adj_bdg=0.0, adj_bds=0.0):
    beta = thermo_beta(temp)
    gval = calc_gval(seq=gseq, temp=temp, dg=dg, adj_bdg37=adj_bdg, adj_bds=adj_bds)
    gb = np.exp(beta * gval)
    return (0.5 * (tile_conc - blocker_conc - gb + np.sqrt((blocker_conc - tile_conc + gb)**2 + 4 * tile_conc * gb))) / tile_conc

def growth_rate(temp, cov_mult, tile_conc=TILE_CONC, gseq=SINGLE_SEQ, gamma=1, kf=1e6, dslat=DS_LAT, order=2, dg=None, tbdg=None, bonds=2, adj_bdg=0.0, adj_bds=0.0):
    """Two-bond growth rate, per second"""
    beta = thermo_beta(temp)
    gval = calc_gval(seq=gseq, temp=temp, dg=dg)
    
    # Calculate probability of correct binding
    pa = pa_full(temp, cov_mult, tile_conc, gseq, dg=dg, adj_bdg=adj_bdg, adj_bds=adj_bds)
    
    # Apply order parameter to probability
    rf = kf * tile_conc * pa**order

    if tbdg is not None:
        rr2 = kf * np.exp(beta * tbdg)
    else:
        rr2 = kf * np.exp(bonds * beta * gval - (bonds-1)*dslat/R_CONST)
    
    return gamma * (rf - rr2)

def assembly_energy(n_tiles, n_bonds, temp, cov_mult, tile_conc=TILE_CONC, gseq=SINGLE_SEQ, dg=None, pba=False, dslat=DS_LAT, adj_bdg=0.0, adj_bds=0.0):
    rt = rt_val(temp)
    gval = calc_gval(seq=gseq, temp=temp, dg=dg)
    pa = pa_full(temp, cov_mult, tile_conc, gseq, dg=dg, adj_bdg=adj_bdg, adj_bds=adj_bds)
    dglat = -(temp + 273.15) * dslat

    if pba:
        bond_energy = n_bonds * gval + (n_bonds - n_tiles) * dglat - rt * n_bonds * np.log(pa)
        tile_chempot = -rt * n_tiles * np.log(tile_conc)
    else:
        bond_energy = n_bonds * gval + (n_bonds - n_tiles) * dglat
        tile_chempot = -rt * n_tiles * (np.log(tile_conc) + 2*np.log(pa))

    return bond_energy + tile_chempot

def square_energy(size, temp, cov_mult, tile_conc=TILE_CONC, gseq=SINGLE_SEQ, dg=None, pba=False, dslat=DS_LAT, adj_bdg=0.0, adj_bds=0.0):
    n_bonds = 2 * size * (size - 1)
    n_tiles = size * size
    return assembly_energy(n_tiles, n_bonds, temp, cov_mult, tile_conc, gseq, dg, pba, dslat, adj_bdg, adj_bds)

def rectangle_energy(width, height, temp, cov_mult, tile_conc=TILE_CONC, gseq=SINGLE_SEQ, dg=None, pba=False, dslat=DS_LAT, tube: None | int = None, adj_bdg=0.0, adj_bds=0.0):
    if tube is not None:
        height = np.minimum(height, tube)
    n_bonds = width * (height - 1) + (width - 1) * height
    if tube is not None:
        n_bonds += width * (height >= tube)
    n_tiles = width * height
    return assembly_energy(n_tiles, n_bonds, temp, cov_mult, tile_conc, gseq, dg, pba, dslat, adj_bdg, adj_bds)

def nuc_rate_rect_mult(temp, mults, pba=False, adj_bdg=0.0, adj_bds=0.0):
    rect_sizes = [(1,1), (1,2), (2,2), (2,3), (3,3), (3,4), (4,4), (4,5), (5,5), (5,6), (6,6), (6,7), 
                (7,7), (7,8), (8,8), (8,9), (9,9), (9,10), 
                (10,10), (10,11), (11,11), (11,12), (12,12), (12,13)]
    rect_sizes = np.array(rect_sizes)
    mults = mults[:,None]
    rect_ntiles = rect_sizes[:,0] * rect_sizes[:,1]

    cngr = rectangle_energy(rect_sizes[:,0], rect_sizes[:,1], temp, mults, pba=pba, adj_bdg=adj_bdg, adj_bds=adj_bds).max(axis=1)
    cnsr = rectangle_energy(rect_sizes[:,0], rect_sizes[:,1], temp, mults, pba=pba, adj_bdg=adj_bdg, adj_bds=adj_bds).argmax(axis=1)
    cpr = np.exp(-cngr) * pa_full(temp, mults[:,0], adj_bdg=adj_bdg, adj_bds=adj_bds)**2 * K_F * TILE_CONC
    return cpr

def nuc_rate_rect_temps(temps, mult, tile_conc=TILE_CONC, pba=False, adj_bdg=0.0, adj_bds=0.0):
    rect_sizes = [(1,1), (1,2), (2,2), (2,3), (3,3), (3,4), (4,4), (4,5), (5,5), (5,6), (6,6), (6,7), 
                (7,7), (7,8), (8,8), (8,9), (9,9), (9,10), 
                (10,10), (10,11), (11,11), (11,12), (12,12), (12,13)]
    temps = temps[:,None]
    rect_sizes = np.array(rect_sizes)
    rect_ntiles = rect_sizes[:,0] * rect_sizes[:,1]

    cngrt = rectangle_energy(rect_sizes[:,0], rect_sizes[:,1], temps, mult, tile_conc, pba=pba, adj_bdg=adj_bdg, adj_bds=adj_bds).max(axis=1)
    cnsrt = rectangle_energy(rect_sizes[:,0], rect_sizes[:,1], temps, mult, tile_conc, pba=pba, adj_bdg=adj_bdg, adj_bds=adj_bds).argmax(axis=1)
    cprt = np.exp(-cngrt) * K_F * tile_conc * pa_full(temps[:,0], mult, adj_bdg=adj_bdg, adj_bds=adj_bds)**2
    return cprt

def nuc_rate_rect(temp, mult, tile_conc=TILE_CONC, pba=False, adj_bdg=0.0, adj_bds=0.0):
    rect_sizes = [(1,1), (1,2), (2,2), (2,3), (3,3), (3,4), (4,4), (4,5), (5,5), (5,6), (6,6), (6,7), 
                (7,7), (7,8), (8,8), (8,9), (9,9), (9,10), 
                (10,10), (10,11), (11,11), (11,12), (12,12), (12,13)]
    rect_sizes = np.array(rect_sizes)
    rect_ntiles = rect_sizes[:,0] * rect_sizes[:,1]

    cngrt = rectangle_energy(rect_sizes[:,0], rect_sizes[:,1], temp, mult, tile_conc, pba=pba, adj_bdg=adj_bdg, adj_bds=adj_bds).max()
    cnsrt = rectangle_energy(rect_sizes[:,0], rect_sizes[:,1], temp, mult, tile_conc, pba=pba, adj_bdg=adj_bdg, adj_bds=adj_bds).argmax()
    cprt = np.exp(-cngrt) * K_F * tile_conc * pa_full(temp, mult, adj_bdg=adj_bdg, adj_bds=adj_bds)**2
    return cprt