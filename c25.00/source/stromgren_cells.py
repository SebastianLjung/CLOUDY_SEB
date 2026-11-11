# How many star-hosting SPHINX cells hold an unresolved Stromgren sphere, using the same Q(H) and Stromgren estimate
# as plot_SPHINXHIIs.py (cell selection):
#   N_ion_max = sqrt(Q V / ((1+Y_He) alpha_B))      atoms the cell's stars can keep ionised in uniform gas at T_ION
#   crit      = N_ion_max / N_target                 Stromgren ionised fraction (> 1: the stars could ionise the whole cell)
#   R_S       = (3 Q / (4 pi (1+Y_He) alpha_B nH^2))^(1/3) = R_out crit^(2/3)   Stromgren radius at the cell's mean nH
# Plots x_HII(SPHINX) / min(crit, 1) and R_S against the smallest cell size, by number of cells and by share of Q(H).
# Run from source/: python stromgren_cells.py
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

HALO_DIR = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/cloudy_haloes/timestep_279/halo_997')
BPASS_FILE = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/CLOUDY_TEST/c25.00/data/bpass_sphinx.ascii')
MODELS_SUBDIR = 'cloudy_models_v2'   # the Cloudy models of plot_SPHINXHIIs.py (results.csv: one row per modelled cell)
OUT = HALO_DIR / 'Stromgren_unresolved.png'
T_ION = 1e4                 # [K] as in plot_SPHINXHIIs.py

pc2cm = 3.086e18
h_cgs = 6.62607e-27
clight = 2.99792458e10
lambda_912 = 911.2671
BPASS_AGE_MIN = 1.0e6
BPASS_LOGZ_MIN, BPASS_LOGZ_MAX = -5.00, -1.40
R_eq_cube = (3 / (4 * np.pi))**(1 / 3)
Y_He = 10**-1.103

plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({'font.size': 12})
plt.rcParams['lines.linewidth'] = 2


# ---- same as plot_SPHINXHIIs.py ----
def load_bpass(path):
    lines = [l for l in Path(path).read_text().split('\n') if not l.startswith('#')]
    n_models, n_wave = int(lines[5]), int(lines[6])
    v = np.array(' '.join(lines[11:]).split(), dtype=float)
    params = v[:2 * n_models].reshape(n_models, 2)
    wave = v[2 * n_models:2 * n_models + n_wave]
    flux = v[2 * n_models + n_wave:2 * n_models + n_wave * (n_models + 1)]
    ages, Zs = np.unique(params[:, 0]), 10**np.unique(params[:, 1])
    sed = flux.reshape(Zs.size, ages.size, n_wave)
    sed = np.concatenate([np.zeros((Zs.size, 1, n_wave)), sed], axis=1)
    return np.concatenate([[0.0], ages]), Zs, wave, sed


def find_host_cells(cell_pos, cell_l, star_pos, k=8):
    _, nearest = cKDTree(cell_pos, leafsize=100).query(star_pos, k=k)
    half_dx = 0.5 * 0.5**cell_l[nearest]
    inside = np.all(np.abs(star_pos[:, None, :] - cell_pos[nearest]) <= half_dx[..., None], axis=2)
    host = nearest[np.arange(len(star_pos)), np.argmax(inside, axis=1)]
    return host, inside.any(axis=1)


def alpha_B(T):
    lam = 315614.0 / T
    return 2.753e-14 * lam**1.5 / (1.0 + (lam / 2.74)**0.407)**2.242


gas, stars = dict(np.load(HALO_DIR / 'gas_cells.npz')), dict(np.load(HALO_DIR / 'stars.npz'))
n_cells = gas['nH'].size
star_age = stars['star_age'].astype(float)
star_Z = stars['star_met'].astype(float)
star_minit = stars['star_minit'].astype(float) * 1e11
star_cell, star_in_cell = find_host_cells(gas['cell_pos'], gas['cell_l'], stars['pos'])

BP_AGES, BP_Z, BP_WAVE, BP_SED = load_bpass(BPASS_FILE)
ion = BP_WAVE <= lambda_912
q_tab = np.trapezoid(BP_SED[..., ion] * BP_WAVE[ion] * 1e-8 / (h_cgs * clight), BP_WAVE[ion], axis=-1)
q_interp = RegularGridInterpolator((np.log10(BP_Z), np.log10(BP_AGES[1:])), np.log10(q_tab[:, 1:]))
logZ = np.clip(np.log10(star_Z), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
logt = np.clip(np.log10(star_age), np.log10(BP_AGES[1]), np.log10(BP_AGES[-1]))
star_Q = 10**q_interp(np.c_[logZ, logt]) * star_minit

Q_cell = np.bincount(star_cell[star_in_cell], weights=star_Q[star_in_cell], minlength=n_cells)
N_star = np.bincount(star_cell[star_in_cell], minlength=n_cells)
dx = gas['cell_dx']
N_target = gas['nH'] * dx**3
N_ion_max = np.sqrt(Q_cell * dx**3 / ((1 + Y_He) * alpha_B(T_ION)))
crit = N_ion_max / N_target
# ---- end of plot_SPHINXHIIs.py part ----

R_S = (3 * Q_cell / (4 * np.pi * (1 + Y_He) * alpha_B(T_ION) * gas['nH']**2))**(1 / 3)    # [cm]
x_S = np.minimum(crit, 1)                       # Stromgren ionised fraction of the cell
dx_min = dx.min()

m = (N_star > 0) & (Q_cell > 0)                 # cells with stars that emit ionising photons
Q_tot = Q_cell[m].sum()
x_ratio = gas['xHII'][m] / x_S[m]
R_S_pc, dx_pc, w = R_S[m] / pc2cm, dx[m] / pc2cm, Q_cell[m] / Q_tot

print(f'{HALO_DIR.name}: {np.sum(N_star > 0)} cells with stars, {m.sum()} with Q(H) > 0 (total {Q_tot:.3e} s^-1); '
      f'smallest cell {dx_min / pc2cm:.2f} pc')
def share(label, s):
    print(f'  {label:52s} {s.sum():8d}  cells {s.mean():6.3f}   Q(H) {w[s].sum():6.3f}')
print('                                                         number  fraction       share')
share('R_S < own cell size dx', R_S_pc < dx_pc)
cloudy = np.isin(np.where(m)[0], pd.read_csv(HALO_DIR / MODELS_SUBDIR / 'results.csv')['cell'].values)   # modelled cells
share(f'CLOUDY models ({MODELS_SUBDIR}), R_S > own dx', cloudy & (R_S_pc > dx_pc))
share(f'CLOUDY models ({MODELS_SUBDIR}), R_S < own dx', cloudy & (R_S_pc < dx_pc))
for q in (0.1, 0.5, 0.9):
    print(f'  {q:.0%} of Q(H) is in cells with R_S below '
          f'{np.interp(q, np.cumsum(w[np.argsort(R_S_pc)]), np.sort(R_S_pc)):.3g} pc')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

bins = np.linspace(-2.5, 3.5, 49)
lr = np.clip(np.log10(np.maximum(x_ratio, 1e-30)), bins[0], bins[-1])
ax1.hist(lr, bins=bins, histtype='step', color='black', label='cells')
ax1.hist(lr, bins=bins, weights=w * m.sum(), histtype='step', color='C1', ls='--', label=r'Q(H)-weighted (× N$_{\rm cells}$)')
ax1.axvline(0, ls=':', color='#606060', lw=1.5)
ax1.set_yscale('log')
ax1.set_xlabel(r'log$_{10}$ x$_{\rm HII}$(SPHINX) / x$_{\rm S}$', size=16)
ax1.set_ylabel('Number of cells', size=16)
ax1.legend(fontsize=10)

bins = np.linspace(np.floor(np.log10(R_S_pc.min())), np.ceil(np.log10(R_S_pc.max())), 61)
lR = np.log10(R_S_pc)
ax2.hist(lR, bins=bins, histtype='step', color='black', label='cells')
ax2.hist(lR, bins=bins, weights=w * m.sum(), histtype='step', color='C1', ls='--', label=r'Q(H)-weighted (× N$_{\rm cells}$)')
ax2.axvline(np.log10(dx_min / pc2cm), ls=':', color='#606060', lw=1.5, label=f'smallest cell {dx_min / pc2cm:.1f} pc')
ax2.set_yscale('log')
ax2.set_xlabel(r'log$_{10}$ R$_{\rm S}$ [pc]', size=16)
ax2.legend(fontsize=10)

fig.tight_layout()
fig.savefig(OUT)
print('saved', OUT)
