# Sends the transmitted spectrum of Balmertest_age0_col18 (rerun at a finer continuum resolution) through a slab of
# neutral hydrogen (N_HI = 1e18 cm^-2, ionised skin kept negligible by diluting the light), to see whether the slab
# rounds the Lyman edge between 912 and ~913 Ang (Lyman series lines crowding towards the limit).
#
# Steps:
#   1. rerun MODEL_IN with `set continuum resolution` = RESOLUTION (outputs in neutral_slab/)
#   2. write its "net trans" continuum as a `table SED` file
#   3. run the neutral slab: same deck lines as MODEL_IN apart from the incident spectrum, its luminosity and the
#      radius (R_SLAB, far enough that the incident LyC can only ionise a thin skin)
#   4. compare the spectra around 912 Ang, with pure photoelectric absorption exp(-N_HI sigma_bf) as reference
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import subprocess
from pathlib import Path

###########################################################
#                      KEY PARAMETERS                     #
###########################################################

MODEL_IN = 'Sasha_eCMa_col18'   # earlier: 'Balmertest_age0_col18'
RESOLUTION = 0.03           # `set continuum resolution` of both runs (0.3 in MODEL_IN: ~0.9 Ang bins at 912 Ang)
R_SLAB_LOG = 21             # log radius [cm] of the neutral slab (MODEL_IN: 17)
N_HI_SLAB_LOG = 17.5       # log neutral column of the slab [cm^-2]
RUN_CLOUDY = True
FINE_RANGE = None           # [Ryd] e.g. (0.74, 1.03): also save the fine continuum (with line absorption) in this
                            # range. With merge 1 over 885-1230 Ang Cloudy grew to 12 GB and stalled, so off for now

SOURCE_DIR = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/CLOUDY_TEST/c25.00/source')
OUT_DIR = 'neutral_slab'    # sub-folder of SOURCE_DIR for both runs
CLOUDY_EXE = SOURCE_DIR / 'cloudy.exe'
MODEL_1 = f'{MODEL_IN}_res{RESOLUTION:g}'
MODEL_2 = f'{MODEL_1}_neutralslab{N_HI_SLAB_LOG}'
PLOT_DIR = SOURCE_DIR.parent / 'cloudy_plots' / MODEL_IN    # with the other plots of the input model

Ryd2Ang = 911.267
CONT_COLUMNS = ['nu', 'incident', 'trans', 'DiffOut', 'net trans', 'reflc', 'total', 'reflin', 'outlin']


###########################################################
#                        FUNCTIONS                        #
###########################################################

def read_cont(path):
    a = np.loadtxt(path, usecols=range(len(CONT_COLUMNS)))
    return {k: a[:, i] for i, k in enumerate(CONT_COLUMNS)}


def energy(nu, nuLnu):
    return np.trapezoid(nuLnu, np.log(nu))


def photons(nu, nuLnu, band=(1, np.inf)):
    m = (nu >= band[0]) & (nu <= band[1])
    return np.trapezoid(nuLnu[m] / (nu[m] * 2.1798724e-11), np.log(nu[m]))


def write_sed_file(E, nuFnu, out_path, comment):
    keep = nuFnu > 0
    E, nuFnu = E[keep], nuFnu[keep]
    with open(out_path, 'w') as f:
        f.write(f'# {comment}\n')
        f.write(f'{E[0]:.6e}  {nuFnu[0]:.6e}  nuFnu\n')
        for e, i in zip(E[1:], nuFnu[1:]):
            f.write(f'{e:.6e}  {i:.6e}\n')


def edit_deck(deck, replace):
    # replace the lines starting with each key of `replace` (each must occur exactly once)
    lines = deck.split('\n')
    for key, new in replace.items():
        idx = [i for i, l in enumerate(lines) if l.startswith(key)]
        assert len(idx) == 1, f'"{key}" found {len(idx)} times'
        lines[idx[0]] = new
    return '\n'.join(lines)


def run(name):
    t0 = time.time()
    env = dict(os.environ, CLOUDY_DATA_PATH='.:+')
    subprocess.run([str(CLOUDY_EXE), '-r', name], cwd=SOURCE_DIR / OUT_DIR, capture_output=True, text=True, env=env)
    out = (SOURCE_DIR / OUT_DIR / f'{name}.out').read_text()
    ok = 'Cloudy exited OK' in out
    print(f"[{'OK' if ok else 'FAILED'}] {name} ({time.time() - t0:.0f} s)")
    print('\n'.join(l for l in out.split('\n') if 'Calculation stopped' in l or 'Cloudy ends' in l)[-600:])
    if not ok:
        print(out[-1500:])
    return ok


###########################################################
#          1. FIRST MODEL AT A FINER RESOLUTION           #
###########################################################

(SOURCE_DIR / OUT_DIR).mkdir(exist_ok=True)
deck_in = (SOURCE_DIR / f'{MODEL_IN}.in').read_text()
table_in = next(l for l in deck_in.split('\n') if l.startswith('table '))
if table_in.startswith('table star'):     # relative to source/, so give the full path (run from OUT_DIR/)
    table_in = f'table star "{SOURCE_DIR.parent}/data/bpass_sphinx.ascii" age=1e6 -1.8   # run from {OUT_DIR}/'
# (a `table SED` file is found through Cloudy's data path, data/SED/, so it is kept as it is)
deck_1 = edit_deck(deck_in, {
    'set punch prefix': f'set punch prefix "{MODEL_1}"',
    'table ': table_in,
    'set continuum resolution': f'set continuum resolution {RESOLUTION:g}',
    'save recombination coefficients': '#save recombination coefficients element hydrogen ".recomb"   # large, not needed here',
    'save hydrogen populations': '#save hydrogen populations ".Hpop"'})
(SOURCE_DIR / OUT_DIR / f'{MODEL_1}.in').write_text(deck_1)
if RUN_CLOUDY and not (SOURCE_DIR / OUT_DIR / f'{MODEL_1}.cont').exists():
    print(f'Running {MODEL_1} ...')
    run(MODEL_1)


###########################################################
#               2-3. NEUTRAL SLAB MODEL                   #
###########################################################

con1 = read_cont(SOURCE_DIR / OUT_DIR / f'{MODEL_1}.cont')
nu = con1['nu']
L_in = energy(nu, con1['net trans'])
sed_name = f'{MODEL_1}_nettrans.sed'
write_sed_file(nu, con1['net trans'], SOURCE_DIR / OUT_DIR / sed_name, f'{MODEL_1} "net trans" continuum')

Q_in = photons(nu, con1['net trans'])
phi = Q_in / (4 * np.pi * 10**(2 * R_SLAB_LOG))
print(f'[check] {MODEL_1}: net trans {L_in:.3e} erg/s, Q(H) {Q_in:.3e} 1/s; bins at 912 Ang '
      f'{np.median(np.diff(np.log(nu))[(nu[1:] > 0.95) & (nu[1:] < 1.05)]) * 912:.3f} Ang')
print(f'[check] slab at r = 1e{R_SLAB_LOG} cm: ionising flux {phi:.2e} 1/cm^2/s, '
      f'ionised skin ~ phi / (n alpha_B) = {phi / (100 * 2.6e-13):.1e} cm^-2 (slab N_HI = 1e{N_HI_SLAB_LOG})')

deck_2 = edit_deck(deck_1, {
    'set punch prefix': f'set punch prefix "{MODEL_2}"',
    'table ': f'table SED "{sed_name}" # net trans of {MODEL_1}',
    'luminosity': f'luminosity total {np.log10(L_in):.6f} # [log erg/s] = all of the net trans of {MODEL_1}',
    'radius': f'radius {R_SLAB_LOG} # far away, so only a thin skin of the slab is ionised',
    'stop column density neutral': f'stop column density neutral {N_HI_SLAB_LOG}'})
if FINE_RANGE is not None:
    # the coarse `save continuum` has no line absorption; the fine continuum (Cloudy's line-resolving mesh) has
    deck_2 = deck_2.replace('save last continuum ".cont"', 'save last continuum ".cont"\n'
                            f'save last fine continuum ".fcont" range {FINE_RANGE[0]} {FINE_RANGE[1]} merge 1', 1)
(SOURCE_DIR / OUT_DIR / f'{MODEL_2}.in').write_text(deck_2)
if RUN_CLOUDY:
    print(f'Running {MODEL_2} ...')
    run(MODEL_2)


###########################################################
#                   4. COMPARISON PLOT                    #
###########################################################

con2 = read_cont(SOURCE_DIR / OUT_DIR / f'{MODEL_2}.cont')
ovr = np.loadtxt(SOURCE_DIR / OUT_DIR / f'{MODEL_2}.ovr', comments='#', ndmin=2)
with open(SOURCE_DIR / OUT_DIR / f'{MODEL_2}.ovr') as f:
    cols = f.readline().lstrip('#').rstrip('\n').split('\t')
o = {k: ovr[:, i] for i, k in enumerate(cols)}
dr = np.diff(np.concatenate(([0], o['depth'])))
print(f"[check] slab: x_HII {o['HII'].min():.2e}-{o['HII'].max():.2e}, N_HI {np.sum(o['hden'] * o['HI'] * dr):.2e}, "
      f"N_HII {np.sum(o['hden'] * o['HII'] * dr):.2e} cm^-2, {len(dr)} zones")
print(f"[check] slab: incident {energy(con2['nu'], con2['incident']):.4e} erg/s (given {L_in:.4e}); "
      f"Q(H) in {photons(con2['nu'], con2['incident']):.3e}, out (net trans) {photons(con2['nu'], con2['net trans']):.3e} 1/s")

# Photoelectric absorption only (hydrogenic cross-section, Verner-like nu^-3 scaling is close enough near the edge)
nu_H = nu / 0.99946         # over the H I edge (0.99946 Ryd, 911.75 Ang)
sigma_bf = np.where(nu_H >= 1, 6.30e-18 * (1.34 * nu_H**-2.99 - 0.34 * nu_H**-3.99), 0)
pure_abs = con1['net trans'] * np.exp(-10**N_HI_SLAB_LOG * sigma_bf)

plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({'font.size': 12})
plt.rcParams['lines.linewidth'] = 2
PLOT_DIR.mkdir(parents=True, exist_ok=True)
lam1, lam2 = Ryd2Ang / nu, Ryd2Ang / con2['nu']
for xlim, tag in [((880, 960), 'wide'), ((905, 925), 'zoom'), ((500, 900), 'LyC')]:
    m1 = (lam1 > xlim[0]) & (lam1 < xlim[1])
    norm = np.median(con1['net trans'][(lam1 > 940) & (lam1 < 960)])
    plt.plot(lam1, con1['net trans'] / norm, c='tab:blue', label=f'{MODEL_IN} (res {RESOLUTION:g}): net trans')
    plt.plot(lam1, pure_abs / norm, c='orange', ls='--', label=r'$\times\ e^{-N_{\rm HI}\sigma_{\rm bf}}$ only')
    plt.plot(lam2, con2['net trans'] / norm, c='purple', label=rf'+ neutral slab $N_{{\rm HI}}=10^{{{N_HI_SLAB_LOG}}}$: net trans')
    plt.plot(lam2, con2['trans'] / norm, c='purple', ls=':', label='+ neutral slab: trans')
    plt.axvline(911.75, c='grey', lw=1, ls=':')
    plt.yscale('log')
    plt.xlim(xlim)
    plt.ylim(1e-4, 30)
    plt.xlabel(r'Wavelength [$\rm{\AA}$]', size=16)
    plt.ylabel(r'$\nu L_\nu$ / (940-960 $\rm{\AA}$)', size=16)
    plt.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'Lyedge_{tag}_{MODEL_2}.jpg', dpi=150)
    plt.clf()
print(f'saved Lyedge_wide/zoom_{MODEL_2}.jpg in {PLOT_DIR}')
