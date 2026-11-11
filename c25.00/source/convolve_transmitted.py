# Takes the transmitted continuum of a first Cloudy model, convolves it with a thermal (Doppler) Gaussian at T_CONV,
# and uses the result as the incident spectrum of a second Cloudy model with otherwise the same input file.
#
# Steps:
#   1. read the first model's `save continuum` file (nu [Ryd], nu L_nu [erg/s])
#   2. convolve nu L_nu in ln(nu) with a Gaussian of width sigma = sqrt(k T_CONV / (MASS_AMU m_H)) / c
#      (a Doppler shift is the same fractional shift at every frequency, and nu L_nu per ln(nu) keeps the energy)
#   3. write it as a `table SED` file and the second model's input: a copy of the first deck with only the incident
#      spectrum (table star -> table SED), its normalisation (luminosity total = that of the convolved spectrum)
#      and the save prefix changed
#   4. run Cloudy and plot the incident, transmitted and convolved spectra of both models
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import subprocess
from pathlib import Path

###########################################################
#                      KEY PARAMETERS                     #
###########################################################

MODEL_IN = 'Balmertest_age0_col18'   # first model (its .in and .cont in SOURCE_DIR)
CONT_COLUMN = 'net trans'   # 'net trans' = attenuated stars + outward diffuse (nebular) continuum; 'trans' = stars only
T_CONV = 1e4                # [K] temperature of the Gaussian
MASS_AMU = 1.0              # mass of the moving particles [amu] (1 = hydrogen)
RUN_CLOUDY = True           # False: only write the .sed and .in files

SOURCE_DIR = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/CLOUDY_TEST/c25.00/source')
CLOUDY_EXE = SOURCE_DIR / 'cloudy.exe'
MODEL_OUT = f'{MODEL_IN}_conv{T_CONV:.0e}K'.replace('+0', '').replace('+', '')   # e.g. Balmertest_age0_col18_conv1e4K
PLOT_DIR = SOURCE_DIR.parent / 'cloudy_plots' / MODEL_OUT

# Constants (cgs)
kB, m_H, clight = 1.380649e-16, 1.6726e-24, 2.998e10
Ryd2Ang = 911.267              # lambda [Ang] = Ryd2Ang / E [Ryd]
Lsun = 3.828e33                # [erg/s]

CONT_COLUMNS = ['nu', 'incident', 'trans', 'DiffOut', 'net trans', 'reflc', 'total', 'reflin', 'outlin']


###########################################################
#                        FUNCTIONS                        #
###########################################################

def read_cont(path):
    # {column: array} from `save continuum` (nu [Ryd], the rest nu L_nu [erg/s] for a model with a luminosity)

    a = np.loadtxt(path, usecols=range(len(CONT_COLUMNS)))
    return {k: a[:, i] for i, k in enumerate(CONT_COLUMNS)}


def gaussian_convolve_lnnu(nu, nuLnu, sigma):
    # Convolve nu L_nu with a Gaussian of width sigma in ln(nu): resample to an even ln(nu) grid fine enough for both
    # the Gaussian and Cloudy's bins, convolve, and average back over each of Cloudy's bins (keeps the energy)

    lnnu = np.log(nu)
    step = min(sigma / 4, np.min(np.diff(lnnu)) / 2)
    grid = np.arange(lnnu[0], lnnu[-1] + step, step)
    y = np.interp(grid, lnnu, nuLnu)
    half = int(np.ceil(5 * sigma / step))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * step / sigma)**2)
    y_conv = np.convolve(y, kernel / kernel.sum(), mode='same')
    # average over the bin around each of Cloudy's points (edges halfway between neighbours)
    edges = np.concatenate(([lnnu[0]], 0.5 * (lnnu[1:] + lnnu[:-1]), [lnnu[-1]]))
    cum = np.concatenate(([0], np.cumsum(y_conv) * step))
    cum_at = np.interp(edges, grid - step / 2, cum[:-1])
    return np.diff(cum_at) / np.diff(edges), grid.size


def energy(nu, nuLnu, band=(0, np.inf)):
    # integral of nu L_nu over ln(nu) within band [Ryd] -> [erg/s]
    m = (nu >= band[0]) & (nu <= band[1])
    return np.trapezoid(nuLnu[m], np.log(nu[m]))


def photons(nu, nuLnu, band=(1, np.inf)):
    # number of photons per second within band [Ryd]: nu L_nu / (h nu) over ln(nu)
    m = (nu >= band[0]) & (nu <= band[1])
    return np.trapezoid(nuLnu[m] / (nu[m] * 2.1798724e-11), np.log(nu[m]))


def write_sed_file(E, nuFnu, out_path, comment):
    # Two-column `table SED` file: energy [Ryd], nu*F_nu; the flux type is named on the first data line
    # (as in plot_SPHINXHIIs.py; points with zero flux are left out, Cloudy treats the SED as zero beyond its ends)

    keep = nuFnu > 0
    E, nuFnu = E[keep], nuFnu[keep]
    with open(out_path, 'w') as f:
        f.write(f'# {comment}\n')
        f.write(f'{E[0]:.6e}  {nuFnu[0]:.6e}  nuFnu\n')
        for e, i in zip(E[1:], nuFnu[1:]):
            f.write(f'{e:.6e}  {i:.6e}\n')


def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f'{m} min {s:02d} s' if m else f'{s} s'


###########################################################
#               CONVOLVE THE FIRST MODEL                  #
###########################################################

t_start = time.time()
con1 = read_cont(SOURCE_DIR / f'{MODEL_IN}.cont')
nu, spec = con1['nu'], con1[CONT_COLUMN]

sigma_v = np.sqrt(kB * T_CONV / (MASS_AMU * m_H))    # [cm/s], 1D velocity dispersion
sigma = sigma_v / clight                              # in ln(nu) (= d lambda / lambda)
bins = np.diff(np.log(nu))
print(f'[check] {MODEL_IN}.cont: {nu.size} points, {nu.min():.2e}-{nu.max():.2e} Ryd; using "{CONT_COLUMN}"')
print(f'[check] Gaussian at T = {T_CONV:.1e} K, {MASS_AMU:g} amu: sigma_v = {sigma_v/1e5:.2f} km/s, '
      f'sigma = {sigma:.2e} in ln(nu) = {sigma * 4861:.3f} Ang at H-beta, {sigma * 912:.4f} Ang at 912 Ang')
print(f'[check] Cloudy bins in ln(nu): median {np.median(bins):.2e} (0.5-2 Ryd: {np.median(bins[(nu[1:] > 0.5) & (nu[1:] < 2)]):.2e}), '
      f'so sigma is {sigma / np.median(bins):.3f} of a bin')

spec_conv, n_grid = gaussian_convolve_lnnu(nu, spec, sigma)
spec_conv[spec <= 0] = 0                               # keep Cloudy's zero-flux ends at zero
L1, L2 = energy(nu, spec), energy(nu, spec_conv)
Q1, Q2 = photons(nu, spec), photons(nu, spec_conv)
print(f'[check] convolved on {n_grid} points: total {L1:.4e} -> {L2:.4e} erg/s ({L2/L1-1:+.2e}), '
      f'Q(H) {Q1:.4e} -> {Q2:.4e} 1/s ({Q2/Q1-1:+.2e})')
dev = np.abs(spec_conv - spec) / np.maximum(spec, 1e-300)
print(f'[check] change per point: median {np.median(dev[spec > 0]):.2e}, max {np.max(dev[spec > 0]):.2e} '
      f'(at {Ryd2Ang / nu[spec > 0][np.argmax(dev[spec > 0])]:.1f} Ang)')


###########################################################
#               SECOND MODEL: SED AND INPUT               #
###########################################################

sed_path = SOURCE_DIR / f'{MODEL_OUT}.sed'
write_sed_file(nu, spec_conv, sed_path,
               f'{MODEL_IN} "{CONT_COLUMN}" convolved with a Gaussian, T = {T_CONV:.1e} K ({sigma_v/1e5:.2f} km/s)')

deck = (SOURCE_DIR / f'{MODEL_IN}.in').read_text()
lines = deck.split('\n')
changes = 0
for i, l in enumerate(lines):
    if l.startswith('set punch prefix'):
        lines[i] = f'set punch prefix "{MODEL_OUT}"'; changes += 1
    elif l.startswith('table star') or l.startswith('table SED'):
        lines[i] = f'table SED "{sed_path.name}" # convolved transmitted spectrum of {MODEL_IN}, T = {T_CONV:.1e} K'; changes += 1
    elif l.startswith('luminosity'):
        lines[i] = f'luminosity total {np.log10(L2):.6f} # [log erg/s] = all of the convolved spectrum'; changes += 1
assert changes == 3, f'expected to change the prefix, incident spectrum and luminosity lines, changed {changes}'
(SOURCE_DIR / f'{MODEL_OUT}.in').write_text('\n'.join(lines))
print(f'[check] wrote {sed_path.name} and {MODEL_OUT}.in (only the prefix, incident spectrum and luminosity differ '
      f'from {MODEL_IN}.in)')


###########################################################
#                       RUN CLOUDY                        #
###########################################################

if RUN_CLOUDY:
    print(f'\nRunning Cloudy for {MODEL_OUT} ...')
    t0 = time.time()
    env = dict(os.environ, CLOUDY_DATA_PATH='.:+')   # `table SED` files are only looked for in the data path
    subprocess.run([str(CLOUDY_EXE), '-r', MODEL_OUT], cwd=SOURCE_DIR, capture_output=True, text=True, env=env)
    out = (SOURCE_DIR / f'{MODEL_OUT}.out').read_text()
    ok = 'Cloudy exited OK' in out
    print(f"[{'OK' if ok else 'FAILED'}] {MODEL_OUT} ({fmt_time(time.time() - t0)})")
    print('\n'.join(l for l in out.split('\n') if 'Calculation stopped' in l or 'Cloudy ends' in l
                    or l.strip().startswith(('W-', 'C-', 'PROBLEM', 'DISASTER')))[:2000])
    if not ok:
        print(out[-1500:])


###########################################################
#                    COMPARISON PLOT                      #
###########################################################

con2_path = SOURCE_DIR / f'{MODEL_OUT}.cont'
if con2_path.exists():
    con2 = read_cont(con2_path)
    Linc2 = energy(con2['nu'], con2['incident'])
    print(f"\n[check] model 2 incident: {Linc2:.4e} erg/s (convolved input {L2:.4e}), "
          f"Q(H) {photons(con2['nu'], con2['incident']):.4e} (input {Q2:.4e})")
    for name, c in [(MODEL_IN, con1), (MODEL_OUT, con2)]:
        print(f"    {name}: Q(H) incident {photons(c['nu'], c['incident']):.3e}, trans {photons(c['nu'], c['trans']):.3e}, "
              f"net trans {photons(c['nu'], c['net trans']):.3e} 1/s; total net trans {energy(c['nu'], c['net trans']):.3e} erg/s")

    plt.rcParams['figure.figsize'] = (7, 5)
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['lines.linewidth'] = 2
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    lam1, lam2 = Ryd2Ang / con1['nu'], Ryd2Ang / con2['nu']
    plt.plot(lam1, con1['incident'], c='grey', label=f'{MODEL_IN}: incident')
    plt.plot(lam1, spec, c='tab:blue', label=f'{MODEL_IN}: {CONT_COLUMN}')
    plt.plot(lam1, spec_conv, c='orange', ls='--', label=f'convolved, T = {T_CONV:.0e} K (model 2 incident)')
    plt.plot(lam2, con2['net trans'], c='purple', label='model 2: net trans')
    plt.plot(lam2, con2['DiffOut'], c='purple', ls=':', label='model 2: nebular (DiffOut)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(200, 1e4)
    top = np.max(con1['incident'][(lam1 > 200) & (lam1 < 1e4)])
    plt.ylim(top * 1e-6, top * 3)
    plt.xlabel(r'Wavelength [$\rm{\AA}$]', size=16)
    plt.ylabel(r'$\nu L_\nu$ [erg s$^{-1}$]', size=16)
    plt.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'SED_convolved_{MODEL_OUT}.jpg', dpi=150)
    print(f'saved {PLOT_DIR / f"SED_convolved_{MODEL_OUT}.jpg"}')

print(f'\nDone: took {fmt_time(time.time() - t_start)}')
