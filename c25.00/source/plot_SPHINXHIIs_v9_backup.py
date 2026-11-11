# Models the unresolved HII regions of a SPHINX20 halo with Cloudy, one model per gas cell that holds stars, and
# measures how much of the stars' LyC escapes each cell.
#
# Steps:
#   1. load the halo's gas cells and stars (from extract_halo_cells.py) and give every star its BPASS Q(H)
#   2. choose the cells to model (CELL SELECTION)
#   3. for each of them: sum the BPASS spectra of its stars (cell.sed) and run Cloudy, changing the density until the
#      model sphere (with the cell's volume) holds the cell's number of H atoms
#   4. save the escaping LyC spectra and escape fractions of each cell, a table of all cells and two comparison plots
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import json
import time
import shutil
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({'font.size': 12})
plt.rcParams['lines.linewidth'] = 2


###########################################################
#                KEY PARAMETERS TO CHANGE                 #
###########################################################

# Which cells get a Cloudy model (see CELL SELECTION)
XHII_TRUST = 1.0            # cells SPHINX has ionised at least this much (1.0: fully, x_HII == 1) ...
N_HI_THIN = 1e15            # [cm^-2] ... or whose neutral column is below this are trusted (f_esc = 1), not modelled
N_HI_MAX = 3e20             # [cm^-2] cells whose neutral column is at least this are not modelled (f_esc = 0)
CRIT_RANGE = (0.5, 2.0)     # near-critical: N_ion_max/N_target in this range. Not used to select, only printed as context
Q_MIN = 8e47                # [1/s] cells whose stars emit fewer ionising photons than this are left out
T_ION = 1e4                 # [K] temperature of the ionised gas in the selection estimates
EXTRA_CELLS = [754692]      # cells modelled even if the rules above leave them out (754692: 1.5% of the halo Q(H),
                            # N_HI_min 1.7e14 cm^-2, tested in filter_tests/thin_754692)

# Running
TEST = False                # True: only run the TEST_N brightest selected cells
TEST_N = 3
N_PRINT = 20                # per-cell printouts only for the N_PRINT brightest cells
N_THREADS = os.cpu_count() - 4    # Cloudy models run at the same time (each needs up to ~1.2 GB of memory)
RERUN_ALL = False           # False: cells with a saved, converged result (cell_result.npz) are not run again
CELL_LIST = None            # text file of cell indices: model exactly these cells instead of the selection (filter_tests.py)
MODELS_SUBDIR = 'cloudy_models_v2'   # sub-folder of HALO_DIR for the cell models and their results. Saved models are
                            # reused without checking how they were made, so after changing the Cloudy setup (deck,
                            # resolution, ...) start a new version: cloudy_models_v3, ...
                            #   v1: continuum resolution 0.1 (some cells from the first run at 1), lines included in some
                            #   v2: continuum resolution 0.5, trusted: x_HII == 1 or N_HI_min < 1e15 cm^-2

# Matching the number of H atoms
USE_SAVED_HDEN = True       # start each cell from the hden saved by earlier runs (hden_converged.json), if any
MAX_ITER = 50               # max number of Cloudy runs per cell
TOL = 0.1                   # converged when N_H(model)/N_H(cell) is within 1 +- TOL
hden_guess_factor = 0.2   # first guess (no saved hden): hden = factor * SPHINX nH
MAX_STEP_DEX = 0.75         # largest change of hden in one step before the target is bracketed [dex]

# Cloudy model and output
Z_ABN = 0.0134              # solar metal mass fraction of solar_GASS10.abn (the abundances the decks start from)
RADIUS_LOG_PC = -4          # inner radius of the Cloudy sphere [log pc]
CONTINUUM_RESOLUTION = 0.5  # Cloudy frequency bin width, as a factor of the default. Bins near 912 Ang are ~3 Ang wide
                            # times this: 1 -> 3 Ang, 0.5 -> 1.5 Ang, 0.1 -> 0.3 Ang
LYC_BAND = (204.0, 911.8)   # LyC band we integrate over [Ang]
SPEC_RANGE = (200.0, 911.8) # wavelength range of the spectra saved for each cell [Ang]

# Paths
HALO_DIR = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/cloudy_haloes/timestep_279/halo_997')
CLOUDY_ROOT = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/CLOUDY_TEST/c25.00')
CLOUDY_EXE = CLOUDY_ROOT / 'source' / 'cloudy.exe'
BPASS_FILE = CLOUDY_ROOT / 'data' / 'bpass_sphinx.ascii'            # Cloudy's own BPASS grid
GAS_FILE = HALO_DIR / 'gas_cells.npz'
STARS_FILE = HALO_DIR / 'stars.npz'
STAR_DIR = HALO_DIR / MODELS_SUBDIR                                 # one sub-folder per cell model
HDEN_FILE = STAR_DIR / 'hden_converged.json'
RESULTS_FILE = STAR_DIR / 'results.csv'
z_snap = json.loads((HALO_DIR / 'halo_info.json').read_text())['redshift']   # (for the CMB)

# units in cgs
pc2cm = 3.086e18
h_cgs = 6.62607e-27         # [erg s]
clight = 2.99792458e10      # [cm/s]
lambda_912 = 911.2671       # 1 Ryd [Ang]

BPASS_AGE_MIN = 1.0e6       # [yr]
BPASS_LOGZ_MIN, BPASS_LOGZ_MAX = -5.00, -1.40
R_eq_cube = (3 / (4 * np.pi))**(1 / 3)   # a sphere with the volume of a cube of side dx has R = 0.620 dx
Y_He = 10**-1.103                        # He/H, as in the decks


###########################################################
#                        FUNCTIONS                        #
###########################################################

# ---- stars and their spectra ----

def load_bpass(path):
    # Cloudy's ascii BPASS grid -> ages [yr], Z [mass fraction], wave [Ang], L_lambda [erg/s/Ang per Msun] as [iZ, iage, iwave]
    # The grid has no age-0 model, so a zero-flux age 0 is added at the front (indices then match the RAMSES all_seds.dat)

    lines = [l for l in Path(path).read_text().split('\n') if not l.startswith('#')]
    n_models, n_wave = int(lines[5]), int(lines[6])
    v = np.array(' '.join(lines[11:]).split(), dtype=float)
    params = v[:2 * n_models].reshape(n_models, 2)          # (age, log Z), age varies fastest
    wave = v[2 * n_models:2 * n_models + n_wave]
    flux = v[2 * n_models + n_wave:2 * n_models + n_wave * (n_models + 1)]
    ages, Zs = np.unique(params[:, 0]), 10**np.unique(params[:, 1])
    sed = flux.reshape(Zs.size, ages.size, n_wave)
    sed = np.concatenate([np.zeros((Zs.size, 1, n_wave)), sed], axis=1)
    return np.concatenate([[0.0], ages]), Zs, wave, sed


def find_host_cells(cell_pos, cell_l, star_pos, k=8):
    # Index of the cell that contains each star, and whether one was found.
    # With AMR the nearest cell centre is not always the containing cell (a small neighbour can be closer than the
    # centre of the big cell the star sits in), so the k nearest centres are checked and the one whose cube
    # (centre +- dx/2 along every axis) holds the star is kept. Positions and dx in box units.

    _, nearest = cKDTree(cell_pos, leafsize=100).query(star_pos, k=k)
    half_dx = 0.5 * 0.5**cell_l[nearest]
    inside = np.all(np.abs(star_pos[:, None, :] - cell_pos[nearest]) <= half_dx[..., None], axis=2)
    found = inside.any(axis=1)
    host = nearest[np.arange(len(star_pos)), np.argmax(inside, axis=1)]
    return host, found


def ionising_photons(L_lambda):
    # Number of H-ionising photons [1/s] of a spectrum L_lambda [erg/s/Ang] on the BPASS wavelength grid (last axis)

    ion = BP_WAVE <= lambda_912
    return np.trapezoid(L_lambda[..., ion] * BP_WAVE[ion] * 1e-8 / (h_cgs * clight), BP_WAVE[ion], axis=-1)


def cell_sed(stars_idx):
    # Summed BPASS spectrum L_lambda [erg/s/Ang] of the given star particles

    logZ = np.clip(np.log10(star_Z[stars_idx]), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
    logt = np.clip(np.log10(np.maximum(star_age[stars_idx], BPASS_AGE_MIN)), np.log10(BP_AGES[1]), np.log10(BP_AGES[-1]))
    return np.sum(10**sed_interp(np.c_[logZ, logt]) * star_minit[stars_idx, None], axis=0)


def alpha_B(T):
    # Case-B H recombination coefficient [cm^3/s] (Hui & Gnedin 1997)

    lam = 315614.0 / T
    return 2.753e-14 * lam**1.5 / (1.0 + (lam / 2.74)**0.407)**2.242


# ---- running Cloudy ----

def write_sed_file(E, nuFnu, out_path, comment):
    # Two-column `table SED` file: energy [Ryd], nu*F_nu; the flux type is named on the first data line

    order = np.argsort(E)
    E, nuFnu = E[order], nuFnu[order]
    keep = nuFnu > 0
    E, nuFnu = E[keep], nuFnu[keep]
    with open(out_path, 'w') as f:
        f.write(f'# {comment}\n')
        f.write(f'{E[0]:.6e}  {nuFnu[0]:.6e}  nuFnu\n')
        for e, i in zip(E[1:], nuFnu[1:]):
            f.write(f'{e:.6e}  {i:.6e}\n')


def make_deck(hden_log, metals_log, logQH, rout_log):
    # Cloudy input for one cell: the cell's summed stellar SED (cell.sed) at the centre of a sphere out to R_out.
    # After changing this, save the models in a new MODELS_SUBDIR version

    return f'''print on
abundances "solar_GASS10.abn" no grains
element abundance helium -1.103
element limit off -8.52703961178103
hden {hden_log:.6f} log
constant gas pressure #constant pressure
set pres ioniz 5000
#set temperature floor 20 linear
turbulence 1km/s
sphere
radius {RADIUS_LOG_PC:g} {rout_log:.4f} log parsecs
set nend 10000
set didz 0.03 # Encourages to make shells thinner
metals {metals_log:.2f} log
table SED "cell.sed"
Q(H) {logQH:.2f}
CMB z={z_snap:.1f}
iterate to convergence
no molecules
no H2 molecule
#no induced processes
stop temperature off
#stop efrac 0.01 linear
set continuum resolution {CONTINUUM_RESOLUTION:g} 
database H-like levels small
database He-like levels small
database stout levels 30 15
database chianti levels 30 15
element aluminium off
element phosphorus off
element chlorine off
element argon off
element potassium off
element calcium off
element titanium off
element chromium off
element manganese off
element cobalt off
element nickel off
element fluorine off
element copper off
element zinc off
save overview last ".ovr"
save last ionization means file = ".ionavr"
save continuum "continuum_00000.out" last units Angstrom no hash
save continuum "continuum.out" last absolute
save incident continuum "continuum_stellar_incident.out" last absolute
'''


def cell_dir(c):
    return STAR_DIR / f'cell_{c:07d}'


def fmt_time(seconds):
    # Time in seconds as e.g. '45 s', '3 min 20 s' or '2 h 05 min'

    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds} s'
    if seconds < 3600:
        return f'{seconds // 60} min {seconds % 60:02d} s'
    return f'{seconds // 3600} h {seconds % 3600 // 60:02d} min'


def run_cloudy(model_dir, deck, verbose):
    # Write model_dir/input.in and run `cloudy.exe input.in` there; returns (True if Cloudy exited OK, run time [s])

    t0 = time.time()
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'input.in').write_text(deck)
    env = dict(os.environ, CLOUDY_DATA_PATH='.:+')   # `table SED` files are only looked for in the data path, so add the run folder
    subprocess.run([str(CLOUDY_EXE), 'input.in'], cwd=model_dir, capture_output=True, text=True, env=env)
    out_path = model_dir / 'input.out'
    ok = out_path.exists() and 'Cloudy exited OK' in out_path.read_text()
    dt = time.time() - t0
    if verbose or not ok:
        print(f"[{'OK' if ok else 'FAILED'}] {model_dir.name} ({fmt_time(dt)})")
    if not ok and out_path.exists():
        print(out_path.read_text()[-1500:])
    return ok, dt


# ---- reading Cloudy's output ----

def read_overview(path):
    # {column: array} from `save overview` (depth [cm], Te [K], hden [cm^-3], HI and HII fractions, ...)

    with open(path) as f:
        cols = f.readline().lstrip('#').rstrip('\n').split('\t')
    d = np.loadtxt(path, comments='#', ndmin=2)
    return {col: d[:, i] for i, col in enumerate(cols)}


def read_continuum(path):
    # {column: array} from `save continuum units Angstrom` (nu L_nu [erg/s])

    cols = ['wave', 'incident', 'trans', 'DiffOut', 'net trans', 'reflc', 'total', 'reflin', 'outlin']
    d = np.loadtxt(path, comments='#', usecols=range(len(cols)), ndmin=2)
    return dict(zip(cols, d.T))


def model_shells(ovr, rout_cm):
    # Zone centres r, thicknesses dr and shell volumes [cm, cm, cm^3]; the shells run from R_in to R_out

    r_in = 10**RADIUS_LOG_PC * pc2cm
    r = r_in + ovr['depth']
    edges = np.concatenate(([r_in], 0.5 * (r[1:] + r[:-1]), [rout_cm]))
    return r, np.diff(edges), 4 * np.pi / 3 * np.diff(edges**3)


def model_atoms(ovr, rout_cm):
    # Total number of H atoms and of ionised H atoms in the model sphere

    r, dr, vol = model_shells(ovr, rout_cm)
    return np.sum(ovr['hden'] * vol), np.sum(ovr['hden'] * ovr['HII'] * vol)


def model_T(ovr, rout_cm):
    # Mean temperature of the model sphere, weighted by the number of H atoms (like a SPHINX cell's T) [K]

    r, dr, vol = model_shells(ovr, rout_cm)
    return np.sum(ovr['Te'] * ovr['hden'] * vol) / np.sum(ovr['hden'] * vol)


def print_model(c, ovr):
    # A few diagnostics of the last model of cell c

    r, dr, vol = model_shells(ovr, rout_cm[c])
    nH, T = ovr['hden'], ovr['Te']
    neutral = ovr['HII'] <= 0.5
    N_H_mod, N_HII_mod = model_atoms(ovr, rout_cm[c])
    stops = re.findall(r'Calculation stopped because (.*?)\.', (cell_dir(c) / 'input.out').read_text())
    T_ion = np.sum(T * nH * vol * ~neutral) / max(np.sum(nH * vol * ~neutral), 1e-300)
    T_neu = np.sum(T * nH * vol * neutral) / np.sum(nH * vol * neutral) if neutral.any() else np.nan
    R_IF = r[np.argmax(neutral)] / rout_cm[c] if neutral.any() else np.nan
    print(f"        x_HII: Cloudy {N_HII_mod / N_H_mod:.4f}, SPHINX {gas['xHII'][c]:.4f}; "
          f"ionisation front at {R_IF:.3f} R_out (nan = none), tau_912 = {ovr['Tau912'][-1] if 'Tau912' in ovr else np.nan:.3g}")
    print(f"        N(HI) = {np.sum(nH * ovr['HI'] * dr):.3e} cm^-2, T ionised = {T_ion:.0f} K, T neutral = {T_neu:.0f} K, "
          f"n_H = {nH.min():.3g}-{nH.max():.3g} cm^-3, {len(r)} zones, stopped: {stops[-1] if stops else '?'}")


# ---- results of one cell ----

def band_integral(wave, nuLnu, band=LYC_BAND):
    # L [erg/s] and N [photons/s] of a nu*L_nu spectrum (wave in Ang), integrated over band [Ang]

    order = np.argsort(wave)
    wave, nuLnu = wave[order], nuLnu[order]
    inside = (wave > band[0]) & (wave < band[1])
    w = np.concatenate(([band[0]], wave[inside], [band[1]]))
    # band edges from the points inside the band only (held flat at the edge): interpolating across 912 Ang would
    # mix in the much brighter non-ionising continuum and swamp a weak escaping spectrum
    f = np.interp(w, wave[inside], nuLnu[inside])
    L = np.trapezoid(f / w, w)                            # L_lambda = nu L_nu / lambda
    N = np.trapezoid(f * 1e-8 / (h_cgs * clight), w)      # L_lambda / (h c / lambda)
    return L, N


def save_cell_result(c):
    # Saves the LyC spectra (in SPEC_RANGE) and escape fractions of a finished cell model to cell_dir(c)/cell_result.npz.
    # Spectra on Cloudy's own wavelength grid, ascending: L_lambda_* [erg/s/Ang] and Q_lambda_* [photons/s/Ang] for
    #   incident: the stars, unattenuated               (Cloudy column 'incident')
    #   stellar:  the stars after the gas               ('trans')
    #   nebular:  emitted by the gas, outwards          ('DiffOut' - 'outlin')
    #   total:    stellar + nebular                     ('net trans' - 'outlin')
    # Escape fractions are photon-number ratios to the incident LyC. The emission lines (outlin) are left out of the
    # output; they still take part in Cloudy's cooling and radiative transfer. Clipped at 0, as the columns are
    # printed to 4 digits.

    ovr = read_overview(cell_dir(c) / 'input.ovr')
    N_H_mod, N_HII_mod = model_atoms(ovr, rout_cm[c])
    con = read_continuum(cell_dir(c) / 'inputcontinuum_00000.out')
    spectra = {'incident': con['incident'], 'stellar': con['trans'],
               'nebular': np.maximum(con['DiffOut'] - con['outlin'], 0),
               'total': np.maximum(con['net trans'] - con['outlin'], 0)}
    order = np.argsort(con['wave'])
    wave = con['wave'][order]
    keep = (wave >= SPEC_RANGE[0]) & (wave <= SPEC_RANGE[1])

    out = {'wave': wave[keep]}
    for name, nuLnu in spectra.items():
        out[f'L_lambda_{name}'] = nuLnu[order][keep] / wave[keep]                 # [erg/s/Ang]
        out[f'Q_lambda_{name}'] = nuLnu[order][keep] * 1e-8 / (h_cgs * clight)    # [photons/s/Ang]
        out[f'L_{name}'], out[f'Q_{name}'] = band_integral(con['wave'], nuLnu)    # integrated over LYC_BAND
    for name in ['stellar', 'nebular', 'total']:
        out[f'f_esc_{name}'] = out[f'Q_{name}'] / out['Q_incident']

    out['Q_H_deck'] = Q_deck[c]                                                       # [1/s]
    out['Q_H_incident'] = band_integral(con['wave'], con['incident'], (1.0, lambda_912))[1]   # should equal Q_H_deck
    out['N_H_ratio'] = N_H_mod / N_target[c]
    out['N_HII_model'] = N_HII_mod
    out['x_HII_model'] = N_HII_mod / N_H_mod
    out['T_model'] = model_T(ovr, rout_cm[c])
    out['hden_log'] = run_hden[c]
    out['lyc_band'] = np.array(LYC_BAND)
    np.savez(cell_dir(c) / 'cell_result.npz', **out)


def load_cell_result(c, check=True):
    # The saved result of a cell model as a dict; with check=True None if there is none or its N_H was not matched
    # within the current TOL (then the cell is run again)

    path = cell_dir(c) / 'cell_result.npz'
    if not path.exists():
        return None
    r = dict(np.load(path))
    if check and abs(r['N_H_ratio'] - 1) > TOL:
        return None
    return r


###########################################################
#                   LOAD SPHINX GAS + STARS               #
###########################################################

t_start = time.time()
gas, stars = dict(np.load(GAS_FILE)), dict(np.load(STARS_FILE))
n_cells = gas['nH'].size

star_age = stars['star_age'].astype(float)               # [yr]
star_Z = stars['star_met'].astype(float)                 # metal mass fraction
star_minit = stars['star_minit'].astype(float) * 1e11    # [Msun] (stored in units of 1e11 Msun)
star_cell, star_in_cell = find_host_cells(gas['cell_pos'], gas['cell_l'], stars['pos'])

print(f'[check] {GAS_FILE.parent.name}: {n_cells} cells, {star_age.size} stars, z = {z_snap:.3f}')
print(f"[check] gas: nH {gas['nH'].min():.2e}-{gas['nH'].max():.2e} cm^-3, T {gas['TK'].min():.0f}-{gas['TK'].max():.2e} K, "
      f"Z {gas['metallicity'].min():.2e}-{gas['metallicity'].max():.2e}, dx {gas['cell_dx'].min()/pc2cm:.2f}-{gas['cell_dx'].max()/pc2cm:.2f} pc")
print(f'[check] stars: age {star_age.min()/1e6:.2f}-{star_age.max()/1e6:.1f} Myr, Z {star_Z.min():.2e}-{star_Z.max():.2e}, '
      f'M_init {star_minit.min():.2e}-{star_minit.max():.2e} Msun')
print(f'[check] {star_in_cell.sum()} stars placed in the cell that contains them, {np.sum(~star_in_cell)} in none '
      f'of the extracted cells (left out)')


###########################################################
#               BPASS: Q(H) OF EVERY STAR                 #
###########################################################

BP_AGES, BP_Z, BP_WAVE, BP_SED = load_bpass(BPASS_FILE)
q_tab = ionising_photons(BP_SED)    # H-ionising photons per second per Msun, for every (Z, age) of the grid

# Interpolate q and whole spectra in (log Z, log age)
q_interp = RegularGridInterpolator((np.log10(BP_Z), np.log10(BP_AGES[1:])), np.log10(q_tab[:, 1:]))
sed_interp = RegularGridInterpolator((np.log10(BP_Z), np.log10(BP_AGES[1:])), np.log10(np.maximum(BP_SED[:, 1:, :], 1e-30)))

logZ = np.clip(np.log10(star_Z), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
logt = np.clip(np.log10(star_age), np.log10(BP_AGES[1]), np.log10(BP_AGES[-1]))
star_Q = 10**q_interp(np.c_[logZ, logt]) * star_minit    # [1/s]
print(f'[check] star log Q(H): {np.log10(star_Q.min()):.2f}-{np.log10(star_Q.max()):.2f}, total {star_Q.sum():.2e} s^-1')


###########################################################
#                      CELL SELECTION                     #
###########################################################
# A cell with stars gets a Cloudy model unless one of these simpler answers is good enough:
#   - SPHINX has (nearly) fully ionised it: x_HII >= XHII_TRUST, or a neutral column below N_HI_THIN (tau_912 < 0.0063):
#     trusted, f_esc = 1. SPHINX also has the photons from other cells and collisional ionisation in hot gas, which a
#     Cloudy model with only the cell's own stars misses. (filter_tests.py: cells with 1e14-1e16 cm^-2 have Cloudy
#     f_esc 0.96-1, while x_HII alone is not a good guide below 0.99)
#   - SPHINX leaves a neutral column of at least N_HI_MAX in it: f_esc = 0 (tau_912 = 6.3e-18 N_HI = 630 at 1e20 cm^-2)
#   - its stars are faint (Q(H) <= Q_MIN): left out, they hold very little of the halo's LyC
#
# The neutral column is that of SPHINX's neutral atoms, N_target (1 - x_HII), all in a thin shell at R_out:
#   N_HI_min = N_target (1 - x_HII) / (4 pi R_out^2): the least radial column they can make (any other arrangement
#              of them gives more)
# Only printed, as context: how many atoms the stars could ionise on their own, for gas of uniform density at T_ION
#   N_ion_max = sqrt(Q V / ((1+Y_He) alpha_B))   (Stromgren: Q = alpha_B n_e n_p V)
#   crit      = N_ion_max / N_target: > 1 the stars could ionise the whole cell, < 1 they cannot;
#               near-critical (crit within CRIT_RANGE) cells have the f_esc most sensitive to the details

N_star = np.bincount(star_cell[star_in_cell], minlength=n_cells)                           # stars per cell
Q_cell = np.bincount(star_cell[star_in_cell], weights=star_Q[star_in_cell], minlength=n_cells)   # summed Q(H) [1/s]
V_cell = gas['cell_dx']**3                                             # [cm^3]
rout_cm = R_eq_cube * gas['cell_dx']                                   # Cloudy sphere with the cell's volume [cm]
N_target = gas['nH'] * V_cell                                          # H atoms in each SPHINX cell

N_ion_max = np.sqrt(Q_cell * V_cell / ((1 + Y_He) * alpha_B(T_ION)))
crit = N_ion_max / N_target
N_HI_min = gas['nH'] *(1 - gas['xHII']) * rout_cm #N_target * (1 - gas['xHII']) / (4 * np.pi * rout_cm**2)

has_stars = N_star > 0
extra = np.isin(np.arange(n_cells), EXTRA_CELLS) & has_stars             # modelled whatever the rules say
ionised = ((gas['xHII'] >= XHII_TRUST) | (N_HI_min < N_HI_THIN)) & ~extra   # trusted, f_esc = 1
thick = (N_HI_min > N_HI_MAX) & ~extra                                 # f_esc = 0
faint = (Q_cell <= Q_MIN) & ~extra
selected = has_stars & ~ionised & ~thick & ~faint
near_critical = (crit > CRIT_RANGE[0]) & (crit < CRIT_RANGE[1])           # only printed

Q_tot = Q_cell.sum()
def line(label, m):
    print(f'  {label:52s} {m.sum():8d}  ({Q_cell[m].sum() / Q_tot:.3f})')
print(f'\nCell selection (share of all stellar Q(H) {Q_tot:.2e} s^-1 in brackets):')
line('cells with stars, of which:', has_stars)
line(f'  - ionised by SPHINX (x_HII >= {XHII_TRUST} or N_HI < {N_HI_THIN:.0e}): f_esc = 1', has_stars & ionised)
line(f'      x_HII >= {XHII_TRUST}', has_stars & (gas['xHII'] >= XHII_TRUST))
line(f'      N_HI < {N_HI_THIN:.0e} cm^-2', has_stars & (N_HI_min < N_HI_THIN))
line(f'      N_HI < {N_HI_THIN:.0e} cm^-2 but x_HII < {XHII_TRUST}', has_stars & (N_HI_min < N_HI_THIN) & (gas['xHII'] < XHII_TRUST))
line(f'  - neutral column > {N_HI_MAX:.0e} cm^-2: f_esc = 0', has_stars & ~ionised & thick)
line(f'  - Q(H) <= Q_MIN = {Q_MIN:.0e} s^-1: left out', has_stars & ~ionised & ~thick & faint)
line('  - modelled with Cloudy, of which:', selected)
line('      stars could ionise the whole cell (N_ion_max >= N_target)', selected & (crit >= 1))
line('      stars cannot, low neutral column', selected & (crit < 1))
line('      EXTRA_CELLS (added by hand)', extra)
print(f'  near-critical cells ({CRIT_RANGE[0]} < N_ion_max/N_target < {CRIT_RANGE[1]}), for context:')
line('      ionised by SPHINX', has_stars & ionised & near_critical)
line(f'      neutral column > {N_HI_MAX:.0e} cm^-2', has_stars & ~ionised & thick & near_critical)
line('      Q(H) <= Q_MIN', has_stars & ~ionised & ~thick & faint & near_critical)
line('      modelled with Cloudy', selected & near_critical)


###########################################################
#                     CELLS TO MODEL                      #
###########################################################
# A cell with a saved, converged cell_result.npz in MODELS_SUBDIR is not run again.

if CELL_LIST is not None:
    selected = np.zeros(n_cells, dtype=bool)
    selected[np.loadtxt(CELL_LIST, dtype=int, ndmin=1)] = True
    print(f'[check] CELL_LIST: modelling the {selected.sum()} cells listed in {CELL_LIST} instead of the selection above')

todo_cells = np.where(selected)[0]
todo_cells = todo_cells[np.argsort(Q_cell[todo_cells])[::-1]]           # brightest first
if TEST:
    todo_cells = todo_cells[:TEST_N]
    print(f'[TEST] only running the {len(todo_cells)} brightest selected cells')
show = set(todo_cells[:N_PRINT])                                        # cells that get per-cell printouts

print(f'\nThe {min(N_PRINT, len(todo_cells))} brightest cells to model:')
top = todo_cells[:N_PRINT]
print(pd.DataFrame({
    'cell': top,
    'dx [pc]': gas['cell_dx'][top] / pc2cm,
    'n_stars': N_star[top],
    'log10 Q(H) [1/s]': np.log10(Q_cell[top]),
    'nH [cm^-3]': gas['nH'][top],
    'x_HII SPHINX': gas['xHII'][top],
    'N_ion_max/N_target': crit[top],
    'N_HI_min [cm^-2]': N_HI_min[top],
}).to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# Keep only the models of the cells selected now in MODELS_SUBDIR, so that copying it never picks up unwanted ones:
# folders of cells that are not selected (any more) are moved to MODELS_SUBDIR/not_selected, and back if a cell is
# selected again (its saved result is then reused). Not done in TEST runs, which only use a few of the selected cells
UNUSED_DIR = STAR_DIR / 'not_selected'
if not TEST:
    keep = {cell_dir(c).name for c in todo_cells}
    n_back = n_out = 0
    for d in UNUSED_DIR.glob('cell_*'):
        if d.name in keep and not (STAR_DIR / d.name).exists():
            shutil.move(d, STAR_DIR / d.name)
            n_back += 1
    for d in STAR_DIR.glob('cell_*'):
        if d.name not in keep:
            UNUSED_DIR.mkdir(exist_ok=True)
            shutil.move(d, UNUSED_DIR / d.name)
            n_out += 1
    print(f'[check] {n_out} folders of cells not selected moved to {STAR_DIR.name}/{UNUSED_DIR.name}, '
          f'{n_back} moved back from it')

saved_results = {} if RERUN_ALL else {c: r for c in todo_cells if (r := load_cell_result(c)) is not None}
run_cells = np.array([c for c in todo_cells if c not in saved_results], dtype=int)
print(f'\n[check] {STAR_DIR.name}: {len(saved_results)} of {len(todo_cells)} cells have a saved, converged result and '
      f'are skipped; {len(run_cells)} cells to run')


###########################################################
#          SUMMED STELLAR SED OF EACH CELL (cell.sed)     #
###########################################################

Q_deck = np.zeros(n_cells)       # Q(H) of each cell's summed SED [1/s]
for c in run_cells:
    idx = np.where((star_cell == c) & star_in_cell)[0]
    L_lam = cell_sed(idx)
    Q_deck[c] = ionising_photons(L_lam)
    cell_dir(c).mkdir(parents=True, exist_ok=True)
    write_sed_file(lambda_912 / BP_WAVE, BP_WAVE * L_lam, cell_dir(c) / 'cell.sed', f'BPASS sum of {idx.size} stars in cell {c}')
off = [c for c in run_cells if abs(Q_deck[c] / Q_cell[c] - 1) > 0.05]       # summed spectrum vs the stars' own Q(H)
if off:
    print(f'[WARNING] {len(off)} cells: Q(H) of the summed stellar spectrum differs by > 5% from the sum of the stars\' '
          f'Q(H), e.g. cell {off[0]}: {Q_deck[off[0]] / Q_cell[off[0]]:.3f}')


###########################################################
#      RUN CLOUDY, ADJUSTING hden UNTIL N_H IS CONSERVED  #
###########################################################
# We trust the number of H atoms and the volume of the SPHINX cell, not its density structure.
# So the Cloudy sphere has the cell's volume (R_out), and hden (density at the inner edge, which sets the
# constant pressure) is changed until the sphere holds the cell's number of H atoms.

hden_log = np.log10(hden_guess_factor * gas['nH'])                        # hden_log of the next run
lo, hi = np.full(n_cells, np.nan), np.full(n_cells, np.nan)               # hden_log known to give too few / too many atoms
lo_ratio, hi_ratio = np.full(n_cells, np.nan), np.full(n_cells, np.nan)   # their N_H ratios
ratio = np.full(n_cells, np.nan)                                          # N_H(model)/N_H(cell) of the last run
run_hden = np.full(n_cells, np.nan)                                       # hden_log of the last run
failed = np.zeros(n_cells, dtype=bool)                                    # Cloudy crashed for this cell
cloudy_time = np.zeros(n_cells)                                           # Cloudy run time summed over iterations [s]
n_runs = np.zeros(n_cells, dtype=int)                                     # number of Cloudy runs

saved_hden = {}
if HDEN_FILE.exists():
    saved_hden = {int(k): v for k, v in json.loads(HDEN_FILE.read_text()).items()}
warm = [c for c in run_cells if c in saved_hden] if USE_SAVED_HDEN else []
print(f'\nStarting density: {len(warm)} of {len(run_cells)} cells start from the density found for them in an earlier '
      f'run ({HDEN_FILE.name}), so should need only a few Cloudy runs; the other {len(run_cells) - len(warm)} start '
      f'from {hden_guess_factor} x SPHINX nH')


def next_hden(c):
    # hden_log for the next run of cell c, from its runs so far.
    # Not bracketed yet (only too-low or only too-high runs): half the step that would match N_H if N_H were
    # proportional to hden, at most MAX_STEP_DEX (N_H can jump when an ionisation front appears, so no bigger steps).
    # Bracketed: between the best too-low and too-high hden, nearer the end that matched N_H better (40/60 split).

    if np.isnan(lo[c]) or np.isnan(hi[c]):
        return hden_log[c] - np.clip(0.5 * np.log10(ratio[c]), -MAX_STEP_DEX, MAX_STEP_DEX)
    if hi_ratio[c] * lo_ratio[c] > 1:               # hi misses by more than lo: lean towards lo
        return 0.6 * lo[c] + 0.4 * hi[c]
    return 0.4 * lo[c] + 0.6 * hi[c]


# Cells with a saved hden continue where their earlier run stopped: from its last hden and N_H ratio, and the
# too-low/too-high hden around the target if it found them (saved since v8), so the first run is a new step rather
# than a repeat of the last one
for c in warm:
    s = saved_hden[c]
    hden_log[c], ratio[c] = s['hden_log'], s['N_H_ratio']
    if s.get('lo') is not None:
        lo[c], lo_ratio[c] = s['lo'], s['lo_ratio']
    if s.get('hi') is not None:
        hi[c], hi_ratio[c] = s['hi'], s['hi_ratio']
    if abs(ratio[c] - 1) > TOL:
        hden_log[c] = next_hden(c)


def save_progress(cells):
    # Save the spectra + escape fractions of these cells (cell_result.npz) and the hden of their last run (HDEN_FILE),
    # with the too-low/too-high hden found so far (None if not found), so an interrupted or unconverged cell continues
    # from there next time

    opt = lambda x: None if np.isnan(x) else float(x)
    for c in cells:
        saved_hden[int(c)] = {'hden_log': float(run_hden[c]), 'N_H_ratio': float(ratio[c]),
                              'converged': bool(abs(ratio[c] - 1) <= TOL),
                              'lo': opt(lo[c]), 'lo_ratio': opt(lo_ratio[c]),
                              'hi': opt(hi[c]), 'hi_ratio': opt(hi_ratio[c])}
        save_cell_result(c)
    STAR_DIR.mkdir(parents=True, exist_ok=True)
    HDEN_FILE.write_text(json.dumps({str(k): v for k, v in sorted(saved_hden.items())}, indent=2))


lock = threading.Lock()        # one cell at a time prints or saves (all cells share hden_converged.json)
cells_done = []
print_every = max(1, round(len(run_cells) / 10))


def converge_cell(c):
    # Run Cloudy for cell c, with a new hden after each run, until N_H is within TOL of the cell's (at most MAX_ITER
    # runs), then save it. Each cell does this on its own, so a slow cell does not hold up the others.

    for it in range(MAX_ITER):
        deck = make_deck(hden_log[c], np.log10(gas['metallicity'][c] / Z_ABN), np.log10(Q_deck[c]),
                         np.log10(rout_cm[c] / pc2cm))
        ok, run_time = run_cloudy(cell_dir(c), deck, False)
        cloudy_time[c] += run_time
        n_runs[c] += 1
        if not ok:
            failed[c] = True
            break

        ovr = read_overview(cell_dir(c) / 'input.ovr')
        ratio[c], run_hden[c] = model_atoms(ovr, rout_cm[c])[0] / N_target[c], hden_log[c]
        if ratio[c] < 1:
            lo[c], lo_ratio[c] = hden_log[c], ratio[c]
        else:
            hi[c], hi_ratio[c] = hden_log[c], ratio[c]

        r_end = (10**RADIUS_LOG_PC * pc2cm + ovr['depth'][-1]) / rout_cm[c]
        with lock:
            if r_end < 0.95:
                print(f'[check] cell {c}: model stopped before R_out ({r_end:.3f} R_out), see the end of input.out')
            if c in show:
                print(f'[check] cell {c}, run {it + 1}: hden {hden_log[c]:.5f} -> N_H(model)/N_H(cell) = {ratio[c]:.3f}')
                print_model(c, ovr)
        if abs(ratio[c] - 1) <= TOL:
            break
        hden_log[c] = next_hden(c)

    with lock:
        cells_done.append(c)
        if failed[c]:
            print(f'[check] cell {c}: Cloudy FAILED at hden {hden_log[c]:.5f} (run {n_runs[c]}), dropped')
        else:
            # saved also when not converged within MAX_ITER, so the spectra are there and the next run
            # starts from its last hden; it is rerun next time
            save_progress([c])
            if abs(ratio[c] - 1) > TOL:
                print(f'[check] cell {c}: NOT converged (N_H ratio {ratio[c]:.3f}) after {n_runs[c]} Cloudy runs, '
                      f'took {fmt_time(cloudy_time[c])}; saved, rerun next time')
        if len(cells_done) % print_every == 0 or len(cells_done) == len(run_cells):
            ok_done = [d for d in cells_done if not failed[d]]
            print(f'    {len(cells_done)} of {len(run_cells)} cells done and saved, after a median of '
                  f'{np.median(n_runs[cells_done]):.0f} Cloudy runs each'
                  + (f'; {len(cells_done) - len(ok_done)} failed' if len(ok_done) < len(cells_done) else '')
                  + f'. {fmt_time(time.time() - t_start)} since the start')


if len(run_cells):
    print(f'\nRunning Cloudy for {len(run_cells)} cells, {min(len(run_cells), N_THREADS)} at a time. Each cell is rerun '
          f'with a new hden until its N_H is within {TOL:.0%} of the SPHINX cell, then saved ...')
    with ThreadPoolExecutor(min(len(run_cells), N_THREADS)) as ex:
        list(ex.map(converge_cell, run_cells))     # list() so that an error inside a cell is raised here
finished = run_cells[~np.isnan(run_hden[run_cells]) & ~failed[run_cells]]
if len(finished):
    print(f'\n[check] saved the spectra of {len(finished)} cells; Cloudy runs per cell: median {np.median(n_runs[finished]):.0f}, '
          f'max {n_runs[finished].max()}; Cloudy time per cell: median '
          f'{fmt_time(np.median(cloudy_time[finished]))}, longest {fmt_time(cloudy_time[finished].max())} '
          f'(cell {finished[np.argmax(cloudy_time[finished])]})')


###########################################################
#                 LyC ESCAPE OF EACH CELL                 #
###########################################################
# Built from the saved cell_result.npz files: cells run now and cells skipped as already done.
# Cells whose last run did not converge within TOL are included but flagged (they are rerun next time).

results = []
for c in todo_cells:
    r = load_cell_result(c, check=False)
    if r is None:
        continue
    if not np.allclose(r['lyc_band'], LYC_BAND):
        print(f"[check] cell {c}: saved escape fractions use the LyC band {r['lyc_band']}, not LYC_BAND = {LYC_BAND}")
    results.append({
        'cell': int(c),
        'n_stars': int(N_star[c]),
        'log10 Q(H) [1/s]': float(np.log10(r['Q_H_deck'])),
        'L_LyC stars [erg/s]': float(r['L_incident']),
        'L_LyC escaping [erg/s]': float(r['L_total']),
        'N_LyC stars [1/s]': float(r['Q_incident']),
        'N_LyC escaping [1/s]': float(r['Q_total']),
        'f_esc stellar': float(r['f_esc_stellar']),
        'f_esc nebular': float(r['f_esc_nebular']),
        'f_esc total': float(r['f_esc_total']),
        'N_H Cloudy / SPHINX': float(r['N_H_ratio']),
        'converged': bool(abs(r['N_H_ratio'] - 1) <= TOL),
        'x_HII Cloudy': float(r['x_HII_model']),
        'x_HII SPHINX': float(gas['xHII'][c]),
        'T Cloudy [K]': float(r['T_model']) if 'T_model' in r else np.nan,
        'T SPHINX [K]': float(gas['TK'][c]),
        'N_ion_max / N_target': float(crit[c]),
        'N_HI_min [cm^-2]': float(N_HI_min[c]),
    })

results = pd.DataFrame(results)
results.to_csv(RESULTS_FILE, index=False)
print(f'\nLyC ({LYC_BAND[0]:g}-{LYC_BAND[1]:g} Ang) of the {len(results)} Cloudy cell models, saved to {RESULTS_FILE}')
print(f'(the {min(N_PRINT, len(results))} brightest shown)')
print(results.head(N_PRINT).to_string(index=False, float_format=lambda x: f'{x:.3g}'))
if len(results):
    print(f"\nAll modelled cells together: f_esc total {results['N_LyC escaping [1/s]'].sum() / results['N_LyC stars [1/s]'].sum():.3f} "
          f"(of {results['N_LyC stars [1/s]'].sum():.2e} LyC photons/s)")


###########################################################
#        PLOTS: SPHINX20 INPUT VS CLOUDY CONVERGED        #
###########################################################
# One point per modelled cell. Cloudy values are averages over the model sphere, weighted by the number of H atoms.
# Grey: cells whose N_H did not converge within TOL.

if len(results):
    conv = results['converged'].values

    # Ionisation fraction, on logit axes: these stretch the values close to 1 (0.9, 0.99, 0.999, ...)
    lims = (0.1, 1 - 1e-4)                       # cells above 1 - 1e-4 are drawn on the top/right edge
    x = np.clip(results['x_HII SPHINX'], *lims)
    y = np.clip(results['x_HII Cloudy'], *lims)
    plt.scatter(x[conv], y[conv], s=3, alpha=0.8, c='purple', label='unresolved cells')
    plt.scatter(x[~conv], y[~conv], s=3, alpha=0.8, c='grey', label='not converged')
    plt.plot(lims, lims, ls=':', color='#606060', lw=1.5, label='1:1')
    plt.xscale('logit')
    plt.yscale('logit')
    plt.xlim(lims)
    plt.ylim(lims)
    ticks = [0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
    plt.xticks(ticks, [f'{t:g}' for t in ticks])
    plt.yticks(ticks, [f'{t:g}' for t in ticks])
    plt.minorticks_off()
    plt.xlabel(r'x$_{\rm{HII}}$ SPHINX20', size=16)
    plt.ylabel(r'x$_{\rm{HII}}$ CLOUDY', size=16)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(STAR_DIR / 'xHII_SPHINX_vs_CLOUDY.png', dpi=150)
    plt.clf()

    # Temperature
    lims = (1e3, 1e6)
    plt.scatter(results['T SPHINX [K]'][conv], results['T Cloudy [K]'][conv], s=3, alpha=0.8, c='purple', label='unresolved cells')
    plt.scatter(results['T SPHINX [K]'][~conv], results['T Cloudy [K]'][~conv], s=3, alpha=0.8, c='grey', label='not converged')
    plt.plot(lims, lims, ls=':', color='#606060', lw=1.5, label='1:1')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(r'T SPHINX20 [K]', size=16)
    plt.ylabel(r'T CLOUDY [K]', size=16)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(STAR_DIR / 'T_SPHINX_vs_CLOUDY.png', dpi=150)
    plt.clf()
    print(f'[check] saved xHII_SPHINX_vs_CLOUDY.png and T_SPHINX_vs_CLOUDY.png to {STAR_DIR}')

print(f'\nDone: {len(run_cells)} cells run, {len(saved_results)} loaded from earlier runs; took {fmt_time(time.time() - t_start)}')
