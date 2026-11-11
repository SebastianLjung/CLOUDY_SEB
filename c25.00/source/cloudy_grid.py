# Builds a grid of Cloudy models of unresolved HII regions, like the cell models of plot_SPHINXHIIs.py but for a
# regular grid of parameters instead of the cells of one halo, so that cells can later be fitted/interpolated
# instead of modelled one by one.
#
# Each model: a BPASS stellar population (one age, one metallicity, scaled to a Q(H)) at the centre of a gas sphere
# with the volume of a cube of side dx, at constant pressure. Like the cell models, hden is changed until the sphere
# holds N_H = nH * dx^3 H atoms (the cell's number of atoms), so the grid axes are:
#   metallicity Z (gas and stars) | stellar age (shape of the SED) | Q(H) (stellar mass/luminosity) | dx (volume) | nH (atoms)
#
# Steps:
#   1. list every combination of the axes (grid_params.csv) and write one BPASS SED per (Z, age) into GRID_DIR/seds
#   2. run Cloudy for each model, changing hden until N_H is matched (as in plot_SPHINXHIIs.py)
#   3. save the escaping LyC spectra and escape fractions of each model (model_result.npz), and collect them in
#      grid_results.csv (one row per model) and grid_results.npz (arrays over the grid axes, for interpolating)
#
# Rerunning the script only runs the models that have no converged result yet, so the grid can be extended by adding
# values to the axes.
import numpy as np
import pandas as pd
import os
import re
import json
import time
import itertools
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import RegularGridInterpolator


###########################################################
#                KEY PARAMETERS TO CHANGE                 #
###########################################################

# Grid axes. Adding values to an axis later only runs the new models
Z_VALUES = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3]       # metal mass fraction, of the gas and of the stars (BPASS)
AGE_MYR = [1, 3, 10, 30]                        # stellar population age [Myr] (sets the shape of the SED)
LOG_Q = [48.0, 48.5, 49.0, 49.5, 50.0, 51.0]    # log10 Q(H) of the stars [1/s] (the stellar mass follows from Q, age, Z)
DX_PC = [10.874, 21.748]                        # cell size [pc] (SPHINX20 AMR levels); sphere R_out = 0.620 dx
NH_VALUES = [0.1, 0.3, 1, 3, 10, 30, 100, 300]  # mean H density of the cell [cm^-3]; N_H = nH dx^3 atoms

# Running
TEST = False                # True: only run the first TEST_N models (in grid order)
TEST_N = 2
N_PRINT = 5                 # per-run printouts only for the first N_PRINT models
N_THREADS = os.cpu_count() - 4    # Cloudy models run at the same time (each needs up to ~1.2 GB of memory)
RERUN_ALL = False           # False: models with a saved, converged result (model_result.npz) are not run again
GRID_NAME = 'grid_v1'       # sub-folder of GRIDS_DIR. Saved models are reused without checking how they were made, so
                            # after changing the Cloudy setup (deck, resolution, ...) start a new version: grid_v2, ...
                            #   v1: continuum resolution 0.5, deck as cloudy_models_v2 of plot_SPHINXHIIs.py

# Matching the number of H atoms
MAX_ITER = 30               # max number of Cloudy runs per model
TOL = 0.02                  # converged when N_H(model)/N_H(target) is within 1 +- TOL
hden_guess_factor = 0.37    # first guess: hden = factor * nH (the median found for the SPHINX cells)
MAX_STEP_DEX = 0.75         # largest change of hden in one step before the target is bracketed [dex]
STUCK_DEX = 1e-3            # bracketed hden range narrower than this but N_H still off: N_H jumps there (an ionisation
                            # front appears), no hden gives the target. The last run is kept, flagged 'stuck'

# Cloudy model and output
Z_ABN = 0.0134              # solar metal mass fraction of solar_GASS10.abn (the abundances the decks start from)
RADIUS_LOG_PC = -4          # inner radius of the Cloudy sphere [log pc]
REDSHIFT = 6.0              # for the CMB (SPHINX20 timestep 279: z = 6.0)
CONTINUUM_RESOLUTION = 0.5  # Cloudy frequency bin width, as a factor of the default. Bins near 912 Ang are ~3 Ang wide
                            # times this: 1 -> 3 Ang, 0.5 -> 1.5 Ang, 0.1 -> 0.3 Ang
LYC_BAND = (204.0, 911.8)   # LyC band we integrate over [Ang]
SPEC_RANGE = (200.0, 911.8) # wavelength range of the spectra saved for each model [Ang]

# Paths
GRIDS_DIR = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/cloudy_grids')
CLOUDY_ROOT = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/CLOUDY_TEST/c25.00')
CLOUDY_EXE = CLOUDY_ROOT / 'source' / 'cloudy.exe'
BPASS_FILE = CLOUDY_ROOT / 'data' / 'bpass_sphinx.ascii'            # Cloudy's own BPASS grid
GRID_DIR = GRIDS_DIR / GRID_NAME
SED_DIR = GRID_DIR / 'seds'                                         # one BPASS SED per (Z, age)
MODEL_DIR = GRID_DIR / 'models'                                     # one sub-folder per model
HDEN_FILE = GRID_DIR / 'hden_converged.json'

# units in cgs
pc2cm = 3.086e18
h_cgs = 6.62607e-27         # [erg s]
clight = 2.99792458e10      # [cm/s]
lambda_912 = 911.2671       # 1 Ryd [Ang]

BPASS_LOGZ_MIN, BPASS_LOGZ_MAX = -5.00, -1.40
R_eq_cube = (3 / (4 * np.pi))**(1 / 3)   # a sphere with the volume of a cube of side dx has R = 0.620 dx
AXES = {'Z': Z_VALUES, 'age [Myr]': AGE_MYR, 'log10 Q(H) [1/s]': LOG_Q, 'dx [pc]': DX_PC, 'nH [cm^-3]': NH_VALUES}


###########################################################
#                        FUNCTIONS                        #
###########################################################

# ---- stellar spectra ----

def load_bpass(path):
    # Cloudy's ascii BPASS grid -> ages [yr], Z [mass fraction], wave [Ang], L_lambda [erg/s/Ang per Msun] as [iZ, iage, iwave]

    lines = [l for l in Path(path).read_text().split('\n') if not l.startswith('#')]
    n_models, n_wave = int(lines[5]), int(lines[6])
    v = np.array(' '.join(lines[11:]).split(), dtype=float)
    params = v[:2 * n_models].reshape(n_models, 2)          # (age, log Z), age varies fastest
    wave = v[2 * n_models:2 * n_models + n_wave]
    flux = v[2 * n_models + n_wave:2 * n_models + n_wave * (n_models + 1)]
    ages, Zs = np.unique(params[:, 0]), 10**np.unique(params[:, 1])
    return ages, Zs, wave, flux.reshape(Zs.size, ages.size, n_wave)


def ionising_photons(L_lambda):
    # Number of H-ionising photons [1/s] of a spectrum L_lambda [erg/s/Ang] on the BPASS wavelength grid (last axis)

    ion = BP_WAVE <= lambda_912
    return np.trapezoid(L_lambda[..., ion] * BP_WAVE[ion] * 1e-8 / (h_cgs * clight), BP_WAVE[ion], axis=-1)


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


def sed_name(Z, age):
    return f'bpass_Z{Z:.1e}_age{age:g}Myr.sed'


# ---- running Cloudy ----

def make_deck(hden_log, metals_log, logQH, rout_log, sed_file):
    # Cloudy input for one model: a BPASS SED at the centre of a sphere out to R_out.
    # The same as make_deck of plot_SPHINXHIIs.py (cloudy_models_v2). After changing this, start a new GRID_NAME version

    return f'''print on
abundances "solar_GASS10.abn" no grains
element abundance helium -1.103
element limit off -8.52703961178103
hden {hden_log:.6f} log
constant gas pressure #constant pressure
set pres ioniz 5000
turbulence 1km/s
sphere
radius {RADIUS_LOG_PC:g} {rout_log:.4f} log parsecs
set nend 10000
set didz 0.03 # Encourages to make shells thinner
metals {metals_log:.2f} log
table SED "{sed_file}"
Q(H) {logQH:.2f}
CMB z={REDSHIFT:.1f}
iterate to convergence
no molecules
no H2 molecule
stop temperature off
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


def model_name(p):
    # Folder name of a model, from its parameters (so that a model keeps its folder when axes are extended)

    return (f"Z{p['Z']:.1e}_age{p['age [Myr]']:g}Myr_logQ{p['log10 Q(H) [1/s]']:.2f}_dx{p['dx [pc]']:.2f}pc_"
            f"nH{p['nH [cm^-3]']:.3g}")


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
    env = dict(os.environ, CLOUDY_DATA_PATH=f'{SED_DIR}/:+')   # `table SED` files are looked for in the data path
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
    # Mean temperature of the model sphere, weighted by the number of H atoms [K]

    r, dr, vol = model_shells(ovr, rout_cm)
    return np.sum(ovr['Te'] * ovr['hden'] * vol) / np.sum(ovr['hden'] * vol)


def model_NHI(ovr, rout_cm):
    # Radial neutral H column from the centre to R_out [cm^-2]

    r, dr, vol = model_shells(ovr, rout_cm)
    return np.sum(ovr['hden'] * ovr['HI'] * dr)


# ---- results of one model ----

def band_integral(wave, nuLnu, band=LYC_BAND):
    # L [erg/s] and N [photons/s] of a nu*L_nu spectrum (wave in Ang), integrated over band [Ang]

    order = np.argsort(wave)
    wave, nuLnu = wave[order], nuLnu[order]
    inside = (wave > band[0]) & (wave < band[1])
    w = np.concatenate(([band[0]], wave[inside], [band[1]]))
    # band edges from the points inside the band only (held flat at the edge), as in plot_SPHINXHIIs.py
    f = np.interp(w, wave[inside], nuLnu[inside])
    return np.trapezoid(f / w, w), np.trapezoid(f * 1e-8 / (h_cgs * clight), w)


def save_model_result(i, status):
    # Saves the LyC spectra (in SPEC_RANGE) and escape fractions of a finished model to model_result.npz, with the
    # same spectra as cell_result.npz of plot_SPHINXHIIs.py (incident, stellar, nebular, total; emission lines left out)

    d = MODEL_DIR / names[i]
    ovr = read_overview(d / 'input.ovr')
    N_H_mod, N_HII_mod = model_atoms(ovr, rout_cm[i])
    con = read_continuum(d / 'inputcontinuum_00000.out')
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

    out['N_H_ratio'] = N_H_mod / N_target[i]
    out['x_HII_model'] = N_HII_mod / N_H_mod
    out['T_model'] = model_T(ovr, rout_cm[i])
    out['N_HI_model'] = model_NHI(ovr, rout_cm[i])
    out['hden_log'] = run_hden[i]
    out['status'] = status                     # 'converged', 'stuck' or 'not converged'
    out['lyc_band'] = np.array(LYC_BAND)
    out['params'] = json.dumps({k: float(v) for k, v in grid.iloc[i].items()})   # the model's grid parameters
    np.savez(d / 'model_result.npz', **out)


def load_model_result(i, check=True):
    # The saved result of model i as a dict; with check=True None if there is none, or its N_H was not matched within
    # the current TOL and it was not stuck (then it is run again)

    path = MODEL_DIR / names[i] / 'model_result.npz'
    if not path.exists():
        return None
    r = dict(np.load(path))
    if check and abs(r['N_H_ratio'] - 1) > TOL and str(r['status']) != 'stuck':
        return None
    return r


###########################################################
#                     GRID OF MODELS                      #
###########################################################

t_start = time.time()
grid = pd.DataFrame(list(itertools.product(*AXES.values())), columns=list(AXES))
names = [model_name(p) for _, p in grid.iterrows()]
n_models = len(grid)

dx_cm = grid['dx [pc]'].values * pc2cm
rout_cm = R_eq_cube * dx_cm                                      # sphere with the cell's volume [cm]
N_target = grid['nH [cm^-3]'].values * dx_cm**3                  # H atoms in the cell
grid['N_H'] = N_target

GRID_DIR.mkdir(parents=True, exist_ok=True)
grid.assign(model=names).to_csv(GRID_DIR / 'grid_params.csv', index_label='index')
print(f'[check] {n_models} models: ' + ' x '.join(f'{len(v)} {k}' for k, v in AXES.items()))
print(f'[check] N_H = {N_target.min():.2e}-{N_target.max():.2e} atoms, R_out = {rout_cm.min() / pc2cm:.2f}-'
      f'{rout_cm.max() / pc2cm:.2f} pc; parameters in {GRID_DIR / "grid_params.csv"}')


###########################################################
#          BPASS SED OF EACH (Z, age) (GRID_DIR/seds)     #
###########################################################
# Single stellar populations, interpolated in (log Z, log age) in Cloudy's BPASS grid. Their normalisation does not
# matter: the deck scales them to the model's Q(H). The stellar mass that gives that Q(H) is saved in grid_params.csv

BP_AGES, BP_Z, BP_WAVE, BP_SED = load_bpass(BPASS_FILE)
sed_interp = RegularGridInterpolator((np.log10(BP_Z), np.log10(BP_AGES)), np.log10(np.maximum(BP_SED, 1e-30)))

SED_DIR.mkdir(parents=True, exist_ok=True)
q_per_msun = {}
for Z, age in itertools.product(Z_VALUES, AGE_MYR):
    logZ = np.clip(np.log10(Z), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
    logt = np.clip(np.log10(age * 1e6), np.log10(BP_AGES[0]), np.log10(BP_AGES[-1]))
    L_lam = 10**sed_interp([logZ, logt])[0]                     # [erg/s/Ang per Msun]
    q_per_msun[Z, age] = ionising_photons(L_lam)
    write_sed_file(lambda_912 / BP_WAVE, BP_WAVE * L_lam, SED_DIR / sed_name(Z, age),
                   f'BPASS SSP, Z = {Z:g}, age = {age:g} Myr, per Msun; Q(H) = {q_per_msun[Z, age]:.3e} 1/s/Msun')
grid['M_star [Msun]'] = [10**p['log10 Q(H) [1/s]'] / q_per_msun[p['Z'], p['age [Myr]']] for _, p in grid.iterrows()]
grid.assign(model=names).to_csv(GRID_DIR / 'grid_params.csv', index_label='index')
print(f'[check] {len(q_per_msun)} SEDs in {SED_DIR.name}: Q(H) per Msun {min(q_per_msun.values()):.2e}-'
      f'{max(q_per_msun.values()):.2e} 1/s; stellar masses {grid["M_star [Msun]"].min():.2e}-'
      f'{grid["M_star [Msun]"].max():.2e} Msun')


###########################################################
#                      MODELS TO RUN                      #
###########################################################

todo = np.arange(n_models)
if TEST:
    todo = todo[:TEST_N]
    print(f'[TEST] only running the first {len(todo)} models')
show = set(todo[:N_PRINT])

saved_results = {} if RERUN_ALL else {i: r for i in todo if (r := load_model_result(i)) is not None}
run_models = np.array([i for i in todo if i not in saved_results], dtype=int)
print(f'\n[check] {GRID_NAME}: {len(saved_results)} of {len(todo)} models have a saved result and are skipped; '
      f'{len(run_models)} models to run')


###########################################################
#      RUN CLOUDY, ADJUSTING hden UNTIL N_H IS MATCHED    #
###########################################################
# As for the SPHINX cells: the sphere has the cell's volume (R_out), and hden (density at the inner edge, which sets
# the constant pressure) is changed until the sphere holds N_H = nH dx^3 atoms.

hden_log = np.log10(hden_guess_factor * grid['nH [cm^-3]'].values)    # hden_log of the next run
lo, hi = np.full(n_models, np.nan), np.full(n_models, np.nan)        # hden_log known to give too few / too many atoms
lo_ratio, hi_ratio = np.full(n_models, np.nan), np.full(n_models, np.nan)   # their N_H ratios
ratio = np.full(n_models, np.nan)                                     # N_H(model)/N_H(target) of the last run
run_hden = np.full(n_models, np.nan)                                  # hden_log of the last run
failed = np.zeros(n_models, dtype=bool)                               # Cloudy crashed for this model
cloudy_time = np.zeros(n_models)                                      # Cloudy run time summed over runs [s]
n_runs = np.zeros(n_models, dtype=int)                                # number of Cloudy runs

saved_hden = {}
if HDEN_FILE.exists():
    saved_hden = json.loads(HDEN_FILE.read_text())
warm = [i for i in run_models if names[i] in saved_hden]
for i in warm:
    hden_log[i] = saved_hden[names[i]]['hden_log']
print(f'\nStarting density: {len(warm)} of {len(run_models)} models start from the density found for them in an '
      f'earlier run ({HDEN_FILE.name}); the other {len(run_models) - len(warm)} start from {hden_guess_factor} x nH')


def next_hden(i):
    # hden_log for the next run of model i, from its runs so far (as in plot_SPHINXHIIs.py).
    # Not bracketed yet: half the step that would match N_H if N_H were proportional to hden, at most MAX_STEP_DEX.
    # Bracketed: between the best too-low and too-high hden, nearer the end that matched N_H better (40/60 split).

    if np.isnan(lo[i]) or np.isnan(hi[i]):
        return hden_log[i] - np.clip(0.5 * np.log10(ratio[i]), -MAX_STEP_DEX, MAX_STEP_DEX)
    if hi_ratio[i] * lo_ratio[i] > 1:               # hi misses by more than lo: lean towards lo
        return 0.6 * lo[i] + 0.4 * hi[i]
    return 0.4 * lo[i] + 0.6 * hi[i]


lock = threading.Lock()        # one model at a time prints or saves (all models share hden_converged.json)
models_done = []
print_every = max(1, round(len(run_models) / 10))


def converge_model(i):
    # Run Cloudy for model i, with a new hden after each run, until N_H is within TOL of the target (at most MAX_ITER
    # runs, or until it is stuck at a jump in N_H), then save it. Each model does this on its own.

    d = MODEL_DIR / names[i]
    p = grid.iloc[i]
    status = 'not converged'
    for it in range(MAX_ITER):
        deck = make_deck(hden_log[i], np.log10(p['Z'] / Z_ABN), p['log10 Q(H) [1/s]'],
                         np.log10(rout_cm[i] / pc2cm), sed_name(p['Z'], p['age [Myr]']))
        ok, run_time = run_cloudy(d, deck, False)
        cloudy_time[i] += run_time
        n_runs[i] += 1
        if not ok:
            failed[i] = True
            break

        ovr = read_overview(d / 'input.ovr')
        ratio[i], run_hden[i] = model_atoms(ovr, rout_cm[i])[0] / N_target[i], hden_log[i]
        if ratio[i] < 1:
            lo[i], lo_ratio[i] = hden_log[i], ratio[i]
        else:
            hi[i], hi_ratio[i] = hden_log[i], ratio[i]

        r_end = (10**RADIUS_LOG_PC * pc2cm + ovr['depth'][-1]) / rout_cm[i]
        with lock:
            if r_end < 0.95:
                print(f'[check] {names[i]}: model stopped before R_out ({r_end:.3f} R_out), see the end of input.out')
            if i in show:
                print(f'[check] {names[i]}, run {it + 1}: hden {hden_log[i]:.5f} -> N_H(model)/N_H(target) = {ratio[i]:.3f}')
        if abs(ratio[i] - 1) <= TOL:
            status = 'converged'
            break
        if hi[i] - lo[i] < STUCK_DEX:
            status = 'stuck'
            break
        hden_log[i] = next_hden(i)

    with lock:
        models_done.append(i)
        if failed[i]:
            print(f'[check] {names[i]}: Cloudy FAILED at hden {hden_log[i]:.5f} (run {n_runs[i]}), dropped')
        else:
            saved_hden[names[i]] = {'hden_log': float(run_hden[i]), 'N_H_ratio': float(ratio[i]), 'status': status}
            save_model_result(i, status)
            HDEN_FILE.write_text(json.dumps(dict(sorted(saved_hden.items())), indent=2))
            if status != 'converged':
                print(f'[check] {names[i]}: {status.upper()} (N_H ratio {ratio[i]:.3f}) after {n_runs[i]} Cloudy runs; saved')
        if len(models_done) % print_every == 0 or len(models_done) == len(run_models):
            print(f'    {len(models_done)} of {len(run_models)} models done and saved, after a median of '
                  f'{np.median(n_runs[models_done]):.0f} Cloudy runs each. {fmt_time(time.time() - t_start)} since the start')


if len(run_models):
    print(f'\nRunning Cloudy for {len(run_models)} models, {min(len(run_models), N_THREADS)} at a time ...')
    with ThreadPoolExecutor(min(len(run_models), N_THREADS)) as ex:
        list(ex.map(converge_model, run_models))     # list() so that an error inside a model is raised here


###########################################################
#                 COLLECT THE GRID RESULTS                #
###########################################################
# grid_results.csv: one row per model (parameters + escape fractions, ...).
# grid_results.npz: the axes, and every quantity as an array over them ([iZ, iage, iQ, idx, inH], NaN where a model is
# missing), e.g. for RegularGridInterpolator; spectra as [..., iwave] on one wavelength grid

SCALARS = ['f_esc_stellar', 'f_esc_nebular', 'f_esc_total', 'Q_incident', 'Q_stellar', 'Q_nebular', 'Q_total',
           'N_H_ratio', 'x_HII_model', 'T_model', 'N_HI_model', 'hden_log']
SPECTRA = ['L_lambda_incident', 'L_lambda_stellar', 'L_lambda_nebular', 'L_lambda_total']

rows, results = [], {}
for i in range(n_models):
    r = load_model_result(i, check=False)
    if r is None:
        continue
    results[i] = r
    rows.append({'model': names[i], **grid.iloc[i].to_dict(), **{k: float(r[k]) for k in SCALARS},
                 'status': str(r['status'])})
table = pd.DataFrame(rows)
table.to_csv(GRID_DIR / 'grid_results.csv', index=False)

shape = tuple(len(v) for v in AXES.values())
arrays = {k: np.full(shape, np.nan) for k in SCALARS}
if results:
    wave = next(iter(results.values()))['wave']
    for k in SPECTRA:
        arrays[k] = np.full(shape + (wave.size,), np.nan)
    arrays['wave'] = wave
    for i, r in results.items():
        idx = np.unravel_index(i, shape)          # grid order is itertools.product of AXES, i.e. C order
        for k in SCALARS:
            arrays[k][idx] = r[k]
        if r['wave'].size == wave.size and np.allclose(r['wave'], wave):
            for k in SPECTRA:
                arrays[k][idx] = r[k]
np.savez(GRID_DIR / 'grid_results.npz', axis_names=np.array(list(AXES)),
         **{f'axis_{n}': np.array(v, dtype=float) for n, v in zip(['Z', 'age_Myr', 'logQ', 'dx_pc', 'nH'], AXES.values())},
         **arrays)

if len(table):
    print(f"\n[check] {len(table)} of {n_models} models saved: {(table['status'] == 'converged').sum()} converged, "
          f"{(table['status'] == 'stuck').sum()} stuck, {(table['status'] == 'not converged').sum()} not converged; "
          f"f_esc total {table['f_esc_total'].min():.2e}-{table['f_esc_total'].max():.3f}")
print(f'\nDone: {len(run_models)} models run, {len(saved_results)} loaded from earlier runs; took '
      f'{fmt_time(time.time() - t_start)}. Results in {GRID_DIR}')
