import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import json
import time
import hashlib
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

TEST = False#True                 # True: only run the TEST_N brightest selected cells
TEST_N = 3
N_PRINT = 20                # per-cell printouts only for the N_PRINT brightest selected cells
USE_SAVED_HDEN = True       # start each cell from the hden saved by earlier runs (hden_converged.json), if any
MAX_ITER = 20               # max number of Cloudy runs per cell to match the number of H atoms
TOL = 0.1                   # converged when N_H(model)/N_H(cell) is within 1 +- TOL
hden_guess_factor = 0.005   # first guess (no saved hden): hden = factor * SPHINX nH

# Which cells get a Cloudy model (see the CELL SELECTION section)
XHII_MAX = 0.95             # never model cells SPHINX has ionised more than this (trusted: f_esc = 1)
N_HI_MAX = 1e20             # [cm^-2] skip cells whose neutral column is at least this (no LyC can escape) ...
CRIT_RANGE = (0.5, 2.0)     # ... unless they are near-critical: N_ion_max/N_target in this range

Q_MIN = 1e48 #1e48                # [1/s] skip cells whose stars emit fewer ionising photons than this

CELL_LIST = None            # path to a text file of cell indices: model exactly these cells instead of the selection
                            # (used by filter_tests.py)
MODELS_SUBDIR = 'cloudy_models'   # sub-folder of HALO_DIR for the cell models and their results

T_ION = 1e4                 # [K] ionised-gas temperature in these estimates, for cells SPHINX has not ionised
T_ION_RANGE = (5e3, 3e4)    # [K] for cells SPHINX has ionised, their own T is used, clipped to this range

Z_ABN = 0.0134              # solar metal mass fraction of solar_GASS10.abn (the abundances the decks start from)
RADIUS_LOG_PC = -4          # inner radius of the Cloudy sphere [log pc]
LYC_BAND = (204.0, 911.8)   # LyC band we integrate over [Ang]
SPEC_RANGE = (200.0, 911.8) # wavelength range of the spectra saved for each cell [Ang]
RERUN_ALL = False           # False: cells with an up-to-date saved result (cell_result.npz) are not run again
KEEP_FESC = (1e-6, 0.99)    # saved results made with an older deck are still kept if their f_esc total is outside this
                            # range (very optically thick/thin cells, insensitive to e.g. the continuum resolution); None = off

# A halo extracted by extract_halo_cells.py; None = the tutorial's galaxy_oct cells
HALO_DIR = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/cloudy_haloes/timestep_279/halo_997')

CLOUDY_ROOT = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/CLOUDY_TEST/c25.00')
PROJECT_ROOT = CLOUDY_ROOT / 'Basu_tutorial'
CLOUDY_EXE = CLOUDY_ROOT / 'source' / 'cloudy.exe'
BPASS_FILE = CLOUDY_ROOT / 'data' / 'bpass_sphinx.ascii'            # Cloudy's own BPASS grid
if HALO_DIR is None:
    GAS_FILE = PROJECT_ROOT / 'data' / 'galaxy_oct' / 'gas_cells.npz'
    STARS_FILE = PROJECT_ROOT / 'data' / 'galaxy_oct' / 'stars.npz'
    STAR_DIR = PROJECT_ROOT / 'runs' / 'halo_HIIs'                  # one sub-folder per cell model
    z_snap = 5.0                                                    # redshift of the snapshot (for the CMB)
else:
    GAS_FILE = HALO_DIR / 'gas_cells.npz'
    STARS_FILE = HALO_DIR / 'stars.npz'
    STAR_DIR = HALO_DIR / MODELS_SUBDIR
    z_snap = json.loads((HALO_DIR / 'halo_info.json').read_text())['redshift']
HDEN_FILE = STAR_DIR / 'hden_converged.json'
RESULTS_FILE = STAR_DIR / 'results.csv'

# units in cgs
pc2cm = 3.086e18
h_cgs = 6.62607e-27         # [erg s]
clight = 2.99792458e10      # [cm/s]
lambda_912 = 911.2671       # 1 Ryd [Ang]
sigma_912 = 6.30e-18        # H photo-ionisation cross-section at 912 Ang [cm^2]

BPASS_AGE_MIN = 1.0e6       # [yr]
BPASS_LOGZ_MIN, BPASS_LOGZ_MAX = -5.00, -1.40
R_eq_cube = (3 / (4 * np.pi))**(1 / 3)   # a sphere with the volume of a cube of side dx has R = 0.620 dx
Y_He = 10**-1.103                        # He/H, as in the decks


###########################################################
#                        FUNCTIONS                        #
###########################################################

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
    # Cloudy input for one cell: the cell's summed stellar SED (cell.sed) at the centre of a sphere out to R_out

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
set continuum resolution 0.1 
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


def run_cloudy_many(cells, decks):
    # Run one Cloudy model per cell in parallel (one per core); returns {cell: True/False (exited OK)}, {cell: run time [s]}

    t0 = time.time()
    n = min(len(cells), os.cpu_count() or 1)
    verbose = len(cells) <= N_PRINT
    print(f'[check] running {len(cells)} CLOUDY models on {n} threads ...')
    with ThreadPoolExecutor(n) as ex:
        runs = list(ex.map(lambda c: run_cloudy(cell_dir(c), decks[c], verbose), cells))
    oks = [ok for ok, dt in runs]
    print(f'[check] CLOUDY finished: {sum(oks)} OK, {len(oks) - sum(oks)} failed, took {fmt_time(time.time() - t0)}')
    return dict(zip(cells, oks)), {c: dt for c, (ok, dt) in zip(cells, runs)}


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


def model_summary(ovr, rout_cm, out_path):
    # A few diagnostics of one model, for printing

    r, dr, vol = model_shells(ovr, rout_cm)
    nH, T = ovr['hden'], ovr['Te']
    ionised = ovr['HII'] > 0.5
    neutral = ~ionised
    stops = re.findall(r'Calculation stopped because (.*?)\.', Path(out_path).read_text())
    return {
        'N_HI': np.sum(nH * ovr['HI'] * dr),                                       # radial column, centre to R_out [cm^-2]
        'N_HII': np.sum(nH * ovr['HII'] * dr),
        'R_IF': r[np.argmax(neutral)] / rout_cm if neutral.any() else np.nan,     # ionisation front [R_out], nan = none
        'T_ion': np.sum(T * nH * vol * ionised) / max(np.sum(nH * vol * ionised), 1e-300),
        'T_neu': np.sum(T * nH * vol * neutral) / np.sum(nH * vol * neutral) if neutral.any() else np.nan,
        'nH_range': f'{nH.min():.3g}-{nH.max():.3g}',
        'tau_912': ovr['Tau912'][-1] if 'Tau912' in ovr else np.nan,
        'zones': len(r),
        'stop': stops[-1] if stops else '?',
    }


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
    # incident (stars, unattenuated), stellar (stars after the gas), nebular (emitted by the gas), total (stellar + nebular)

    ovr = read_overview(cell_dir(c) / 'input.ovr')
    N_H_mod, N_HII_mod = model_atoms(ovr, rout_cm[c])
    con = read_continuum(cell_dir(c) / 'inputcontinuum_00000.out')
    # Cloudy adds the outward line emission (outlin: He I, He II, metals) into DiffOut and net trans; the saved spectra
    # are continuum only (clipped at 0, as the columns are printed to 4 digits). The lines still take part in Cloudy's
    # cooling and radiative transfer, they are only left out of the output
    con['DiffOut cont'] = np.maximum(con['DiffOut'] - con['outlin'], 0)
    con['net trans cont'] = np.maximum(con['net trans'] - con['outlin'], 0)
    order = np.argsort(con['wave'])
    wave = con['wave'][order]
    keep = (wave >= SPEC_RANGE[0]) & (wave <= SPEC_RANGE[1])

    out = {'wave': wave[keep]}
    for name, col in SPECTRA.items():
        nuLnu = con[col][order][keep]
        out[f'L_lambda_{name}'] = nuLnu / wave[keep]                    # [erg/s/Ang]
        out[f'Q_lambda_{name}'] = nuLnu * 1e-8 / (h_cgs * clight)       # [photons/s/Ang]
        out[f'L_{name}'], out[f'Q_{name}'] = band_integral(con['wave'], con[col])   # integrated over LYC_BAND
    out['f_esc_stellar'] = out['Q_stellar'] / out['Q_incident']
    out['f_esc_nebular'] = out['Q_nebular'] / out['Q_incident']
    out['f_esc_total'] = out['Q_total'] / out['Q_incident']

    out['Q_H_deck'] = Q_deck[c]                                                       # [1/s]
    out['Q_H_incident'] = band_integral(con['wave'], con['incident'], (1.0, lambda_912))[1]   # should equal Q_H_deck
    out['N_H_ratio'] = N_H_mod / N_target[c]
    out['N_HII_model'] = N_HII_mod
    out['x_HII_model'] = N_HII_mod / N_H_mod
    out['T_model'] = model_T(ovr, rout_cm[c])
    out['hden_log'] = run_hden[c]
    out['lyc_band'] = np.array(LYC_BAND)
    out['deck_id'] = DECK_ID
    np.savez(cell_dir(c) / 'cell_result.npz', **out)


def load_cell_result(c, check=True):
    # The saved result of a cell model as a dict; with check=True None if there is none or it is out of date
    # (made with a different Cloudy deck, unless its f_esc is outside KEEP_FESC, or N_H not matched within the current TOL)

    path = cell_dir(c) / 'cell_result.npz'
    if not path.exists():
        return None
    r = dict(np.load(path))
    keep_old = KEEP_FESC is not None and not (KEEP_FESC[0] < r['f_esc_total'] < KEEP_FESC[1])
    if check and ((str(r['deck_id']) != DECK_ID and not keep_old) or abs(r['N_H_ratio'] - 1) > TOL):
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
print(f"[check] 'metals' offsets of the gas cells: {np.log10(gas['metallicity'].min()/Z_ABN):+.2f} "
      f"to {np.log10(gas['metallicity'].max()/Z_ABN):+.2f} dex")


###########################################################
#               BPASS: Q(H) OF EVERY STAR                 #
###########################################################

BP_AGES, BP_Z, BP_WAVE, BP_SED = load_bpass(BPASS_FILE)
print(f'[check] BPASS grid: SED {BP_SED.shape} [Z, age, wave], ages {BP_AGES[1]:.1e}-{BP_AGES[-1]:.1e} yr')

q_tab = ionising_photons(BP_SED)    # H-ionising photons per second per Msun, for every (Z, age) of the grid
print(f'log q(H) per Msun at Z=0.02: 1 Myr {np.log10(q_tab[10, 1]):.2f}, 10 Myr {np.log10(q_tab[10, 11]):.2f}, '
      f'100 Myr {np.log10(q_tab[10, 21]):.2f}')

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
# A cell gets a Cloudy model if its stars make an unresolved HII region from which some, but not all, LyC escapes.
# With the cell's H atoms (N_target) spread over the Cloudy sphere (volume V = dx^3):
#  - N_ion_max = sqrt(Q V / ((1+Y_He) alpha_B)) is the most H atoms the stars can keep ionised in V
#    (recombinations Q = alpha_B n_e n_p V_ion; packing the gas any other way only adds recombinations).
#  - N_ion_max >= N_target: the stars can ionise the whole cell -> HII region at least as big as the cell
#    (density bounded): not an unresolved HII region, SPHINX's own radiative transfer handles it.
#  - otherwise at least N_target - N_ion_max neutral atoms lie between the stars and R_out, i.e. a neutral column of at
#    least N_HI_min = (N_target - N_ion_max) / (4 pi R_out^2), whatever the density structure. Above N_HI_MAX
#    no LyC gets out (tau_912 = 6.3e-18 N_HI = 630 at 1e20 cm^-2), so a Cloudy model is not needed.
#  - Near-critical cells (N_ion_max/N_target within CRIT_RANGE: the stars can only just, or only just not, ionise the
#    cell) are modelled: there escape switches between none and nearly all, and only Cloudy can tell which.
#    These estimates use alpha_B at T_ION, or at SPHINX's own T for cells SPHINX has ionised (clipped to T_ION_RANGE,
#    so very hot supernova gas does not make alpha_B tiny); near the critical point a 25% change in alpha_B matters.
#  - Cells SPHINX has ionised (x_HII > XHII_MAX) are never modelled, near-critical or not: SPHINX also has the photons
#    from other cells and collisional ionisation in hot gas, which a Cloudy model with only the cell's own stars misses.

N_star = np.bincount(star_cell[star_in_cell], minlength=n_cells)                           # stars per cell
Q_cell = np.bincount(star_cell[star_in_cell], weights=star_Q[star_in_cell], minlength=n_cells)   # summed Q(H) [1/s]
V_cell = gas['cell_dx']**3                                             # [cm^3]
rout_cm = R_eq_cube * gas['cell_dx']                                   # Cloudy sphere with the cell's volume [cm]
N_target = gas['nH'] * V_cell                                          # H atoms in each SPHINX cell

ionised = gas['xHII'] > XHII_MAX
T_est = np.where(ionised, np.clip(gas['TK'], *T_ION_RANGE), T_ION)     # [K] ionised-gas temperature for alpha_B
N_ion_max = np.sqrt(Q_cell * V_cell / ((1 + Y_He) * alpha_B(T_est)))
crit = N_ion_max / N_target                                            # > 1: the stars can ionise the whole cell
RS_over_Rout = crit**(2 / 3)                                           # Stromgren radius / R_out at the cell's mean density
N_HI_min = np.maximum(N_target - N_ion_max, 0) / (4 * np.pi * rout_cm**2)

has_stars = N_star > 0
density_bounded = crit >= 1
thick = N_HI_min > N_HI_MAX
faint = Q_cell <= Q_MIN
near_critical = (crit > CRIT_RANGE[0]) & (crit < CRIT_RANGE[1])
partial_escape = ~ionised & ~density_bounded & ~thick                  # LyC can partly escape (low-column cells)
selected = has_stars & ~ionised & (near_critical | partial_escape) & ~faint

Q_tot = Q_cell.sum()
def line(label, m):
    print(f'  {label:44s} {m.sum():8d}  ({Q_cell[m].sum() / Q_tot:.3f})')
print(f'\nCell selection (share of all stellar Q(H) {Q_tot:.2e} s^-1 in brackets):')
line('cells with stars', has_stars)
line(f'already ionised (x_HII > {XHII_MAX}, trusted)', has_stars & ionised)
not_ionised = has_stars & ~ionised
line('not ionised, of which:', not_ionised)
line(f'  - near-critical ({CRIT_RANGE[0]} < N_ion_max/N_target < {CRIT_RANGE[1]})', not_ionised & near_critical)
other = not_ionised & ~near_critical
line('  - HII region > cell (density bounded)', other & density_bounded)
line(f'  - neutral column > {N_HI_MAX:.0e} cm^-2', other & ~density_bounded & thick)
line('  - partial escape (low column)', other & partial_escape)
line(f'Q(H) <= Q_MIN = {Q_MIN:.0e} s^-1 (removed)', not_ionised & (near_critical | partial_escape) & faint)
line('= selected for Cloudy', selected)

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
    'R_S / R_out': RS_over_Rout[top],
    'N_HI_min [cm^-2]': N_HI_min[top],
}).to_string(index=False, float_format=lambda x: f'{x:.3g}'))


###########################################################
#          CELLS ALREADY DONE IN EARLIER RUNS             #
###########################################################
# A cell with an up-to-date cell_result.npz is not run again. DECK_ID identifies the Cloudy settings: it is a short
# hash of the deck text (with placeholder values for the per-cell numbers), so changing any line of the deck
# (pressure law, floor, didz, ...) makes all saved results out of date and those cells are run again.

SPECTRA = {'incident': 'incident', 'stellar': 'trans', 'nebular': 'DiffOut cont', 'total': 'net trans cont'}   # name: column
DECK_ID = hashlib.md5(make_deck(0.0, 0.0, 0.0, 0.0).encode()).hexdigest()[:12]

saved_results = {} if RERUN_ALL else {c: r for c in todo_cells if (r := load_cell_result(c)) is not None}
run_cells = np.array([c for c in todo_cells if c not in saved_results], dtype=int)
n_old_deck = sum(str(r['deck_id']) != DECK_ID for r in saved_results.values())
print(f'\n[check] deck id {DECK_ID}: {len(saved_results)} of {len(todo_cells)} selected cells have an up-to-date saved '
      f'result and are skipped ({n_old_deck} of them from an older deck, kept as f_esc is outside KEEP_FESC = {KEEP_FESC}); '
      f'{len(run_cells)} cells to run')


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
    if c in show:
        print(f'[check] cell {c}: {idx.size} stars, Q from summed SED / sum of star Q = {Q_deck[c] / Q_cell[c]:.3f}')


###########################################################
#      RUN CLOUDY, ADJUSTING hden UNTIL N_H IS CONSERVED  #
###########################################################
# We trust the number of H atoms and the volume of the SPHINX cell, not its density structure.
# So the Cloudy sphere has the cell's volume (R_out), and hden (density at the inner edge, which sets the
# constant pressure) is changed until the sphere holds the cell's number of H atoms.
# Once a too-low and a too-high hden are known, the next guess sits between them, nearer the end that
# matched N_H better (40/60 split).

hden_log = np.log10(hden_guess_factor * gas['nH'])
lo, hi = np.full(n_cells, np.nan), np.full(n_cells, np.nan)               # hden_log known to give too few / too many atoms
lo_ratio, hi_ratio = np.full(n_cells, np.nan), np.full(n_cells, np.nan)   # their N_H ratios
ratio = np.full(n_cells, np.nan)                                          # N_H(model)/N_H(cell) of the last run
run_hden = np.full(n_cells, np.nan)                                       # hden_log of the last run
failed = np.zeros(n_cells, dtype=bool)                                    # Cloudy crashed for this cell
cloudy_time = np.zeros(n_cells)                                           # Cloudy run time summed over iterations [s]
n_runs = np.zeros(n_cells, dtype=int)                                     # number of Cloudy runs

saved_hden = {}
if USE_SAVED_HDEN and HDEN_FILE.exists():
    saved_hden = {int(k): v for k, v in json.loads(HDEN_FILE.read_text()).items()}
    n_warm = 0
    for c in run_cells:
        if c in saved_hden:
            hden_log[c] = saved_hden[c]['hden_log']
            n_warm += 1
            if c in show:
                print(f"[check] cell {c}: first guess hden {hden_log[c]:.5f} from {HDEN_FILE.name} "
                      f"(last N_H ratio {saved_hden[c]['N_H_ratio']:.3f}, converged={saved_hden[c]['converged']})")
    print(f'[check] {n_warm} of {len(run_cells)} cells to run start from a saved hden')


def save_progress(cells):
    # Save the spectra + escape fractions of these cells (cell_result.npz) and the hden of their last run (HDEN_FILE),
    # so that an interrupted run keeps them

    for c in cells:
        saved_hden[int(c)] = {'hden_log': float(run_hden[c]), 'N_H_ratio': float(ratio[c]),
                              'converged': bool(abs(ratio[c] - 1) <= TOL)}
        save_cell_result(c)
    STAR_DIR.mkdir(parents=True, exist_ok=True)
    HDEN_FILE.write_text(json.dumps({str(k): v for k, v in sorted(saved_hden.items())}, indent=2))


todo = list(run_cells)
for it in range(MAX_ITER):
    if not todo:
        break
    decks = {c: make_deck(hden_log[c], np.log10(gas['metallicity'][c] / Z_ABN), np.log10(Q_deck[c]),
                          np.log10(rout_cm[c] / pc2cm)) for c in todo}
    ok, run_time = run_cloudy_many(todo, decks)

    for c in todo:
        cloudy_time[c] += run_time[c]
        n_runs[c] += 1
        if not ok[c]:
            print(f'[check] iter {it}: cell {c}: Cloudy FAILED at hden {hden_log[c]:.5f}, dropped')
            failed[c] = True
            continue

        ovr = read_overview(cell_dir(c) / 'input.ovr')
        N_H_mod, N_HII_mod = model_atoms(ovr, rout_cm[c])
        ratio[c], run_hden[c] = N_H_mod / N_target[c], hden_log[c]
        r_end = 10**RADIUS_LOG_PC * pc2cm + ovr['depth'][-1]
        if r_end / rout_cm[c] < 0.95:
            print(f'[check] cell {c}: model stopped before R_out ({r_end / rout_cm[c]:.3f} R_out), see the end of input.out')

        if c in show:
            s = model_summary(ovr, rout_cm[c], cell_dir(c) / 'input.out')
            print(f'[check] iter {it}: cell {c}: hden {hden_log[c]:.5f} -> N_H(model)/N_H(cell) = {ratio[c]:.3f}, '
                  f'last zone at {r_end / rout_cm[c]:.3f} R_out')
            print(f"        ionised fraction of H atoms: Cloudy {N_HII_mod / N_H_mod:.4f}, SPHINX {gas['xHII'][c]:.4f}")
            print(f"        N(HI) = {s['N_HI']:.3e} cm^-2, N(HII) = {s['N_HII']:.3e} cm^-2, "
                  f"ionisation front at {s['R_IF']:.3f} R_out (nan = none), tau_912 = {s['tau_912']:.3g}")
            print(f"        T ionised = {s['T_ion']:.0f} K, T neutral = {s['T_neu']:.0f} K, n_H = {s['nH_range']} cm^-3, "
                  f"{s['zones']} zones, stopped: {s['stop']}")

        if ratio[c] < 1:
            lo[c], lo_ratio[c] = hden_log[c], ratio[c]
        else:
            hi[c], hi_ratio[c] = hden_log[c], ratio[c]

    # Cells that converged in this iteration are saved straight away
    converged_now = [c for c in todo if not failed[c] and abs(ratio[c] - 1) <= TOL]
    save_progress(converged_now)
    for c in converged_now:
        print(f'[check] cell {c}: converged (N_H ratio {ratio[c]:.3f}) after {n_runs[c]} Cloudy runs, '
              f'took {fmt_time(cloudy_time[c])}; saved')

    todo = [c for c in todo if not failed[c] and abs(ratio[c] - 1) > TOL]
    done = run_cells[~np.isnan(ratio[run_cells])]
    print(f'[check] iter {it}: {len(done) - len(todo)} cells converged, {len(todo)} still running, {failed.sum()} failed; '
          f'median |log10 N_H ratio| of the running ones = {np.median(np.abs(np.log10(ratio[todo]))) if todo else 0:.3f}; '
          f'{fmt_time(time.time() - t_start)} since the start')
    for c in todo:
        if np.isnan(lo[c]) or np.isnan(hi[c]):          # not bracketed yet: step towards the target (at most 0.5 dex)
            hden_log[c] -= 0.5 * np.clip(np.log10(ratio[c]), -1, 1)
        elif hi_ratio[c] * lo_ratio[c] > 1:             # hi misses by more than lo: lean towards lo
            hden_log[c] = 0.6 * lo[c] + 0.4 * hi[c]
        else:                                           # lo misses by more than hi: lean towards hi
            hden_log[c] = 0.4 * lo[c] + 0.6 * hi[c]
    if not todo:
        break

if len(run_cells):
    print(f'[check] N_H converged in {it + 1} iterations; {len(todo)} cells not converged, {failed.sum()} failed')

# Converged cells were saved as they converged; now also save the cells that did not converge within MAX_ITER
# (their last run, so the spectra are there and the next run starts from their last hden)
save_progress(todo)
for c in todo:
    print(f'[check] cell {c}: NOT converged (N_H ratio {ratio[c]:.3f}) after {n_runs[c]} Cloudy runs, '
          f'took {fmt_time(cloudy_time[c])}; saved, rerun next time')
finished = run_cells[~np.isnan(run_hden[run_cells]) & ~failed[run_cells]]
print(f'[check] saved hden to {HDEN_FILE} and the spectra of {len(finished)} cells to cell_*/cell_result.npz')
if len(finished):
    print(f'[check] Cloudy time per cell: median {fmt_time(np.median(cloudy_time[finished]))}, '
          f'longest {fmt_time(cloudy_time[finished].max())} (cell {finished[np.argmax(cloudy_time[finished])]}); '
          f'Cloudy part took {fmt_time(time.time() - t_start)} since the start')


###########################################################
#                 LyC ESCAPE OF EACH CELL                 #
###########################################################
# Cloudy continuum columns: incident (stars, unattenuated), trans (stars, after the gas),
# DiffOut (emitted by the gas, outwards: continuum + lines), net trans = trans + DiffOut, outlin (the lines alone).
# Escape fractions are photon-number ratios to the incident stellar LyC:
#   stellar: N(trans)/N(incident),  nebular: N(DiffOut)/N(incident),  total: N(net trans)/N(incident)
# with the lines (outlin) subtracted from DiffOut and net trans

# The table is built from the saved cell_result.npz files: cells run now and cells skipped as already done.
# Cells whose last run did not converge within TOL are included but flagged (they are rerun next time).

def saved_T_fallback(c):
    # Mean T of a cell whose cell_result.npz predates T_model: from its last Cloudy run (input.ovr) instead,
    # nan if that cannot be read (e.g. a run of it is in progress)
    try:
        return float(model_T(read_overview(cell_dir(c) / 'input.ovr'), rout_cm[c]))
    except (KeyError, ValueError, OSError):
        return np.nan


results = []
for c in todo_cells:
    r = load_cell_result(c, check=False)
    if r is None:
        continue
    if c in show and c in run_cells:
        print(f"[check] cell {c}: incident Q(H) / deck Q(H) = {r['Q_H_incident'] / r['Q_H_deck']:.3f}")
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
        'N_HII Cloudy / SPHINX': float(r['N_HII_model'] / (gas['nH'][c] * gas['xHII'][c] * V_cell[c])),
        'x_HII Cloudy': float(r['x_HII_model']),
        'x_HII SPHINX': float(gas['xHII'][c]),
        'T Cloudy [K]': float(r['T_model']) if 'T_model' in r else saved_T_fallback(c),
        'T SPHINX [K]': float(gas['TK'][c]),
        'N_HI_min estimate [cm^-2]': float(N_HI_min[c]),
        'run now': bool(c in run_cells),
    })

results = pd.DataFrame(results)
results.to_csv(RESULTS_FILE, index=False)
print(f'\nLyC ({LYC_BAND[0]:g}-{LYC_BAND[1]:g} Ang) of the {len(results)} Cloudy cell models, saved to {RESULTS_FILE}')
print(f'(the {min(N_PRINT, len(results))} brightest shown)')
print(results.head(N_PRINT).to_string(index=False, float_format=lambda x: f'{x:.3g}'))
if len(results):
    print(f"\nAll modelled cells together: f_esc stellar {np.sum(results['f_esc stellar'] * results['N_LyC stars [1/s]']) / results['N_LyC stars [1/s]'].sum():.3f}, "
          f"total {results['N_LyC escaping [1/s]'].sum() / results['N_LyC stars [1/s]'].sum():.3f} "
          f"(of {results['N_LyC stars [1/s]'].sum():.2e} LyC photons/s)")



###########################################################
#        PLOTS: SPHINX20 INPUT VS CLOUDY CONVERGED        #
###########################################################
# One point per modelled cell. Cloudy values are averages over the model sphere, weighted by the number of H atoms.
# Grey: cells whose N_H did not converge within TOL.

if len(results):
    conv = results['converged'].values

    # Ionisation fraction
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='1:1')
    ax.scatter(results['x_HII SPHINX'][conv], results['x_HII Cloudy'][conv], s=10, c='C0', label='converged')
    ax.scatter(results['x_HII SPHINX'][~conv], results['x_HII Cloudy'][~conv], s=10, c='0.6', label='not converged')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x_{\rm HII}$ SPHINX20')
    ax.set_ylabel(r'$x_{\rm HII}$ CLOUDY')
    ax.legend()
    fig.tight_layout()
    fig.savefig(STAR_DIR / 'xHII_SPHINX_vs_CLOUDY.png', dpi=150)
    plt.close(fig)

    # Temperature
    T_lim = [0.8 * min(results['T SPHINX [K]'].min(), results['T Cloudy [K]'].min()),
             1.25 * max(results['T SPHINX [K]'].max(), results['T Cloudy [K]'].max())]
    fig, ax = plt.subplots()
    ax.plot(T_lim, T_lim, 'k--', lw=1, label='1:1')
    ax.scatter(results['T SPHINX [K]'][conv], results['T Cloudy [K]'][conv], s=10, c='C0', label='converged')
    ax.scatter(results['T SPHINX [K]'][~conv], results['T Cloudy [K]'][~conv], s=10, c='0.6', label='not converged')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(T_lim)
    ax.set_ylim(T_lim)
    ax.set_xlabel('T SPHINX20 [K]')
    ax.set_ylabel('T CLOUDY [K]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(STAR_DIR / 'T_SPHINX_vs_CLOUDY.png', dpi=150)
    plt.close(fig)
    print(f'[check] saved xHII_SPHINX_vs_CLOUDY.png and T_SPHINX_vs_CLOUDY.png to {STAR_DIR}')

print(f'\nDone: {len(run_cells)} cells run, {len(saved_results)} loaded from earlier runs; took {fmt_time(time.time() - t_start)}')
