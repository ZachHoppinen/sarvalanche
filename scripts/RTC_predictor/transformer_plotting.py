"""
Sarvalanche model evaluation: single-track scene comparison
Runs inference on a diverse subset of single-track scenes and produces
spatial maps + statistical summary figures.
"""

import os
import re
import warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
SCENE_DIR  = Path('/Users/zmhoppinen/Documents/sarvalanche/local/data/scene_cache')
FIG_DIR    = Path('/Users/zmhoppinen/Documents/sarvalanche/local/figures')
WEIGHTS    = Path('/Users/zmhoppinen/Documents/sarvalanche/src/sarvalanche/ml/weights/sar_transformer_best.pth')

STRIDE     = 4
BATCH_SIZE = 128
DEVICE     = 'mps'

# One file per unique zone, preferring 2020
SCENES = [
    'GNFAC_Bridger_Range__track100__2020-12.nc',
    'BTAC_Salt_River_and_Wyoming_Ranges__track100__2020-12.nc',
    'BTAC_Snake_River_Range__track100__2020-12.nc',
    'BTAC_Tetons__track100__2020-12.nc',
    'BTAC_Togwotee_Pass__track100__2020-12.nc',
    'CAIC_CAIC_zone__track151__2020-12.nc',
    'FAC_Flathead_Range_&_Glacier_NP__track122__2020-12.nc',
    'FAC_Swan_Range__track122__2020-12.nc',
    'FAC_Whitefish_Range__track122__2020-12.nc',
    'GNFAC_Cooke_City__track100__2020-12.nc',
    'GNFAC_Island_Park__track100__2020-12.nc',
    'GNFAC_Lionhead_Area__track100__2020-12.nc',
    'GNFAC_Northern_Gallatin_Range__track100__2020-12.nc',
    'GNFAC_Northern_Madison_Range__track122__2020-12.nc',
    'GNFAC_Southern_Gallatin_Range__track100__2020-12.nc',
    'GNFAC_Southern_Madison_Range__track100__2020-12.nc',
    'SNFAC_Banner_Summit__track20__2020-12.nc',
    'SNFAC_Galena_Summit_&_Eastern_Mtns__track20__2020-12.nc',
    'SNFAC_Sawtooth_&_Western_Smoky_Mtns__track93__2020-12.nc',
    'SNFAC_Soldier_&_Wood_River_Valley_Mtns__track20__2020-12.nc',
    'UAC_Moab__track49__2020-12.nc',
    'UAC_Ogden__track100__2020-12.nc',
    'UAC_Provo__track100__2020-12.nc',
    'UAC_Salt_Lake__track100__2020-12.nc',
    'UAC_Uintas__track100__2020-12.nc',
]

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = dict(
    bg      = '#0d1117',
    panel   = '#161b22',
    text    = '#e6edf3',
    muted   = '#8b949e',
    accent  = '#58a6ff',
    good    = '#3fb950',
    warn    = '#d29922',
    bad     = '#f85149',
    cmap_sar = 'RdBu_r',
    cmap_err = 'RdBu',
    cmap_sig = 'YlOrRd',
)

plt.rcParams.update({
    'figure.facecolor':  STYLE['bg'],
    'axes.facecolor':    STYLE['panel'],
    'axes.edgecolor':    STYLE['muted'],
    'axes.labelcolor':   STYLE['text'],
    'xtick.color':       STYLE['muted'],
    'ytick.color':       STYLE['muted'],
    'text.color':        STYLE['text'],
    'grid.color':        STYLE['muted'],
    'grid.alpha':        0.2,
    'font.family':       'monospace',
    'font.size':         9,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def scene_label(fname):
    """Human-readable label from filename."""
    name = fname.replace('.nc', '')
    # strip year
    name = re.sub(r'__\d{4}-\d{2}$', '', name)
    # strip track
    name = re.sub(r'__track\d+', '', name)
    # strip underscores
    return name.replace('_', ' ').replace('&', '&')


def flatten_valid(arr):
    """Return 1-D finite values from any array."""
    flat = np.asarray(arr).ravel()
    return flat[np.isfinite(flat)]


def run_scene(fname, model):
    """Load scene, run inference, return dict of arrays."""
    from sarvalanche.ml.inference import prep_dataset_for_inference, predict_with_sweeping_fast
    path = SCENE_DIR / fname
    ds   = xr.open_dataset(path)
    data = prep_dataset_for_inference(ds['VV'], ds['VH'])

    mu, sigma = predict_with_sweeping_fast(
        model, data[:-1],
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        use_fp16=True,
        min_valid_fraction=0.01,
        device=DEVICE,
    )

    # data is xarray DataArray (T, bands, H, W) = (T, 2, H, W)
    # use .values to get numpy, then index safely
    data_np = np.asarray(data)           # (T, 2, H, W)
    obs     = data_np[-1, 0]             # last timestep, VV band → (H, W)

    mu_np  = np.asarray(mu)
    sig_np = np.asarray(sigma)

    # predict_with_sweeping_fast returns (2, H, W) — take VV channel [0]
    if mu_np.ndim == 3:  mu_np  = mu_np[0]
    if sig_np.ndim == 3: sig_np = sig_np[0]

    print(f'    obs={obs.shape} mu={mu_np.shape} sigma={sig_np.shape}')

    err = mu_np - obs                   # (H, W)

    return dict(fname=fname, label=scene_label(fname),
                obs=obs, mu=mu_np, sigma=sig_np, err=err)


# ── Per-scene spatial figure ──────────────────────────────────────────────────
def plot_scene(res, out_dir):
    obs, mu, sigma, err = res['obs'], res['mu'], res['sigma'], res['err']
    label = res['label']

    vmin_sar, vmax_sar = -30, 5
    v_err = max(abs(np.nanpercentile(err, 2)), abs(np.nanpercentile(err, 98)))
    v_err = min(v_err, 10)

    valid_sig = flatten_valid(sigma)
    mask = np.isfinite(obs) & np.isfinite(mu)
    paired_obs = obs[mask]
    paired_mu  = mu[mask]
    paired_err = err[mask]

    valid_obs = paired_obs  # for histograms
    valid_mu  = paired_mu
    valid_err = paired_err

    mae  = np.mean(np.abs(paired_err))
    rmse = np.sqrt(np.mean(paired_err**2))
    corr = np.corrcoef(paired_obs, paired_mu)[0, 1] if len(paired_obs) > 1 else float('nan')
    mean_sig = np.mean(valid_sig)

    fig = plt.figure(figsize=(18, 9), facecolor=STYLE['bg'])
    fig.suptitle(label, color=STYLE['text'], fontsize=13, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.05, right=0.97, top=0.92, bottom=0.08)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

    def imshow(ax, data, cmap, vmin, vmax, title, symmetric=False):
        if symmetric:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=9, color=STYLE['accent'])
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                     orientation='vertical').ax.tick_params(colors=STYLE['muted'], labelsize=7)
        return im

    imshow(axes[0], obs,   STYLE['cmap_sar'], vmin_sar, vmax_sar, 'Observed VV (dB)')
    imshow(axes[1], mu,    STYLE['cmap_sar'], vmin_sar, vmax_sar, 'Predicted μ (dB)')
    imshow(axes[2], sigma, STYLE['cmap_sig'], 0, min(np.nanpercentile(sigma, 99), 5), 'Predicted σ (dB)')
    imshow(axes[3], err/sigma,   STYLE['cmap_err'], -v_err, v_err, 'Error (μ-obs)/σ (dB)', symmetric=True)

    # Row 2: histograms + scatter
    ax_obs_hist = axes[4]
    ax_err_hist = axes[5]
    ax_sig_hist = axes[6]
    ax_scatter  = axes[7]

    bins = 60
    ax_obs_hist.hist(valid_obs, bins=bins, color=STYLE['accent'], alpha=0.7, density=True, label='obs')
    ax_obs_hist.hist(valid_mu,  bins=bins, color=STYLE['good'],   alpha=0.5, density=True, label='μ')
    ax_obs_hist.set_title('VV distribution', fontsize=9, color=STYLE['accent'])
    ax_obs_hist.legend(fontsize=7, framealpha=0.3)
    ax_obs_hist.set_xlabel('dB')

    ax_err_hist.hist(valid_err, bins=bins, color=STYLE['warn'], alpha=0.8, density=True)
    ax_err_hist.axvline(0, color=STYLE['text'], lw=1, ls='--')
    ax_err_hist.set_title(f'Error dist  MAE={mae:.3f}  RMSE={rmse:.3f}', fontsize=9, color=STYLE['accent'])
    ax_err_hist.set_xlabel('μ − obs (dB)')

    ax_sig_hist.hist(valid_sig, bins=bins, color=STYLE['bad'], alpha=0.8, density=True)
    ax_sig_hist.set_title(f'σ dist  mean={mean_sig:.3f}', fontsize=9, color=STYLE['accent'])
    ax_sig_hist.set_xlabel('σ (dB)')

    # Scatter: obs vs mu (subsample for speed)
    # Both are already 1-D finite arrays; find common valid pixels
    obs_flat = flatten_valid(obs)
    mu_flat  = flatten_valid(mu)
    # Pair them by masking the 2D arrays together
    mask = np.isfinite(obs) & np.isfinite(mu)
    paired_obs = obs[mask]
    paired_mu  = mu[mask]
    n = min(20000, len(paired_obs))
    idx = np.random.choice(len(paired_obs), n, replace=False)
    _o, _m = paired_obs[idx], paired_mu[idx]
    n2 = len(_o)
    ax_scatter.scatter(_o[:n2], _m[:n2], s=1, alpha=0.3, color=STYLE['accent'], rasterized=True)
    lims = [min(vmin_sar, np.nanmin(_o[:n2])), max(vmax_sar, np.nanmax(_o[:n2]))]
    ax_scatter.plot(lims, lims, color=STYLE['warn'], lw=1, ls='--')
    ax_scatter.set_xlim(lims); ax_scatter.set_ylim(lims)
    ax_scatter.set_title(f'obs vs μ  r={corr:.3f}', fontsize=9, color=STYLE['accent'])
    ax_scatter.set_xlabel('obs'); ax_scatter.set_ylabel('μ')

    safe = res['fname'].replace('&', 'and').replace(' ', '_').replace('.nc', '')
    out_path = out_dir / f'{safe}.png'
    fig.savefig(out_path, dpi=110, bbox_inches='tight', facecolor=STYLE['bg'])
    plt.close(fig)
    return out_path


# ── Summary figure ────────────────────────────────────────────────────────────
def plot_summary(all_results, out_dir):
    labels = [r['label'] for r in all_results]
    maes   = []
    rmses  = []
    corrs  = []
    mean_sigs = []
    biases = []

    for r in all_results:
        obs  = r['obs']
        mu   = r['mu']
        err  = r['err']
        sig  = flatten_valid(r['sigma'])
        mask = np.isfinite(obs) & np.isfinite(mu)
        p_obs = obs[mask]
        p_mu  = mu[mask]
        p_err = err[mask]
        maes.append(np.mean(np.abs(p_err)))
        rmses.append(np.sqrt(np.mean(p_err**2)))
        biases.append(np.mean(p_err))
        corrs.append(np.corrcoef(p_obs, p_mu)[0, 1] if len(p_obs) > 1 else float('nan'))
        mean_sigs.append(np.mean(sig))

    maes       = np.array(maes)
    rmses      = np.array(rmses)
    corrs      = np.array(corrs)
    mean_sigs  = np.array(mean_sigs)
    biases     = np.array(biases)

    n_scenes = len(labels)
    short_labels = [l.split(' ')[-1] if len(l) > 25 else l for l in labels]
    y = np.arange(n_scenes)

    fig, axes = plt.subplots(1, 5, figsize=(22, max(6, n_scenes * 0.4 + 2)),
                             facecolor=STYLE['bg'])
    fig.suptitle('Model evaluation — single-track scenes 2020', fontsize=13,
                 color=STYLE['text'], fontweight='bold', y=1.01)

    def bar_h(ax, vals, color, title, xlabel, ref=None):
        bars = ax.barh(y, vals, color=color, alpha=0.8, height=0.65)
        if ref is not None:
            ax.axvline(ref, color=STYLE['warn'], lw=1.2, ls='--', label=f'ref={ref:.2f}')
        ax.set_yticks(y)
        ax.set_yticklabels(short_labels, fontsize=7.5)
        ax.set_title(title, color=STYLE['accent'], fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(axis='x', alpha=0.25)
        # annotate
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', ha='left', fontsize=6.5,
                    color=STYLE['muted'])
        return ax

    bar_h(axes[0], maes,      STYLE['accent'], 'MAE (dB)',    'dB',  ref=float(np.median(maes)))
    bar_h(axes[1], rmses,     STYLE['warn'],   'RMSE (dB)',   'dB',  ref=float(np.median(rmses)))
    bar_h(axes[2], corrs,     STYLE['good'],   'Correlation', 'r',   ref=float(np.median(corrs)))
    bar_h(axes[3], mean_sigs, STYLE['bad'],    'Mean σ (dB)', 'dB',  ref=float(np.median(mean_sigs)))
    bar_h(axes[4], biases,    '#a371f7',       'Bias μ−obs',  'dB',  ref=0.0)

    # remove y-labels from non-first axes
    for ax in axes[1:]:
        ax.set_yticklabels([])

    plt.tight_layout()
    out_path = out_dir / 'summary_stats.png'
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=STYLE['bg'])
    plt.close(fig)
    print(f'  → Summary saved: {out_path}')

    # Print table
    print(f'\n{"Scene":<45} {"MAE":>6} {"RMSE":>6} {"r":>6} {"σ̄":>6} {"bias":>7}')
    print('─' * 80)
    for lbl, mae, rmse, corr, msig, bias in zip(labels, maes, rmses, corrs, mean_sigs, biases):
        print(f'{lbl:<45} {mae:6.3f} {rmse:6.3f} {corr:6.3f} {msig:6.3f} {bias:7.3f}')
    print('─' * 80)
    print(f'{"MEDIAN":<45} {np.median(maes):6.3f} {np.median(rmses):6.3f} '
          f'{np.nanmedian(corrs):6.3f} {np.median(mean_sigs):6.3f} {np.median(biases):7.3f}')

    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    from sarvalanche.ml.inference import load_model

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    scene_fig_dir = FIG_DIR / 'scenes'
    scene_fig_dir.mkdir(exist_ok=True)

    print(f'Loading model from {WEIGHTS}')
    model = load_model(str(WEIGHTS))
    print('Model loaded.\n')

    all_results = []
    for i, fname in enumerate(SCENES):
        path = SCENE_DIR / fname
        if not path.exists():
            print(f'  [{i+1}/{len(SCENES)}] SKIP (not found): {fname}')
            continue
        print(f'  [{i+1}/{len(SCENES)}] {fname}')
        try:
            res = run_scene(fname, model)
            all_results.append(res)
            out = plot_scene(res, scene_fig_dir)
            print(f'          → {out.name}  '
                  f'MAE={np.mean(np.abs(flatten_valid(res["err"]))):.3f}  '
                  f'r={np.corrcoef(flatten_valid(res["obs"])[:1000], flatten_valid(res["mu"])[:1000])[0,1]:.3f}')
        except Exception as e:
            print(f'          ERROR: {e}')

    if all_results:
        print(f'\nGenerating summary figure for {len(all_results)} scenes...')
        plot_summary(all_results, FIG_DIR)
    else:
        print('No results — check paths.')


if __name__ == '__main__':
    np.random.seed(0)
    main()