"""Generate CNFAIC CNN experiment visualization figures."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import re
from pathlib import Path
from datetime import datetime

BASEDIR = Path('/Users/zmhoppinen/Documents/sarvalanche')
FIGDIR = BASEDIR / 'local/cnfaic/figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# ============================================================
# Figure 1: Model Comparison Overview
# ============================================================
def fig1_model_comparison():
    models = ['3stage', 'human_only', '2stage']
    metrics = {
        'Det@0.2': [28/85, 35/85, 1/85],
        'Det@0.5': [25/85, 12/85, 1/85],
        'F1@0.2':  [0.129, 0.166, 0.029],
        'F1@0.5':  [0.126, 0.112, 0.033],
    }

    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for i, (label, vals) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[i], edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Held-out Validation (87 obs after 2026-01-28)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / 'model_comparison_overview.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved model_comparison_overview.png')


# ============================================================
# Figure 2: D-size Detection Rates
# ============================================================
def fig2_dsize_detection():
    dsizes = ['D1', 'D1.5', 'D2', 'D2.5', 'D3']
    three_stage = [0.43, 0.14, 0.23, 0.13, 0.00]
    human_only  = [0.29, 0.43, 0.46, 0.60, 0.43]
    # Sample counts (approximate from 87 total obs)
    sample_counts = [7, 7, 35, 15, 7]

    x = np.arange(len(dsizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, three_stage, width, label='3stage @0.2', color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x + width/2, human_only, width, label='human_only @0.2', color='#FF9800', edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Avalanche Size (D-size)')
    ax.set_ylabel('Detection Rate')
    ax.set_title('Detection Rate by Avalanche Size (threshold > 0.2)')
    ax.set_xticks(x)
    labels = [f'{d}\n(n={n})' for d, n in zip(dsizes, sample_counts)]
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / 'dsize_detection_rates.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved dsize_detection_rates.png')


# ============================================================
# Figure 3: Temporal Detection Profile
# ============================================================
def fig3_temporal_profile():
    # Parsed from experiment_run.log season summaries
    dates_str = [
        '2025-11-09', '2025-11-11', '2025-11-21', '2025-11-23',
        '2025-12-03', '2025-12-05', '2025-12-15', '2025-12-17',
        '2025-12-27', '2025-12-29', '2026-01-08', '2026-01-10',
        '2026-02-01', '2026-02-03', '2026-02-13', '2026-02-15',
        '2026-02-25', '2026-02-27', '2026-03-09', '2026-03-11',
    ]
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_str]

    # >0.5 pixel counts from log
    three_stage_px = [
        79176, 77401, 54951, 54413, 37048, 36401, 258844, 256562,
        19192, 19007, 11448, 11797, 9094, 9127, 324009, 326398,
        291332, 287987, 17272, 30732,
    ]
    two_stage_px = [
        30738, 30549, 17206, 17115, 18542, 17954, 212498, 210798,
        7444, 7512, 209, 242, 2034, 1990, 237116, 238957,
        231166, 231472, 3952, 6850,
    ]
    human_only_px = [
        35729, 33562, 17804, 16008, 12071, 12046, 127189, 124920,
        10254, 7522, 2452, 2049, 4516, 5545, 107737, 118871,
        89365, 74405, 2290, 6135,
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, np.array(three_stage_px)/1000, 'o-', color='#2196F3', label='3stage', linewidth=2, markersize=5)
    ax.plot(dates, np.array(human_only_px)/1000, 's-', color='#FF9800', label='human_only', linewidth=2, markersize=5)
    ax.plot(dates, np.array(two_stage_px)/1000, '^-', color='#4CAF50', label='2stage', linewidth=2, markersize=5)

    # Annotate big events
    for d_str in ['2025-12-15', '2026-02-13', '2026-02-25']:
        d = datetime.strptime(d_str, '%Y-%m-%d')
        ax.axvline(d, color='red', alpha=0.2, linestyle='--', linewidth=1)

    ax.set_xlabel('SAR Date')
    ax.set_ylabel('Pixels > 0.5 (thousands)')
    ax.set_title('Season Temporal Profile: Pixels > 0.5 per SAR date')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / 'temporal_detection_heatmap.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved temporal_detection_heatmap.png')


# ============================================================
# Figures 4-6: Example Date Probability Maps
# ============================================================
def fig456_example_dates():
    print('Loading inference datasets...')
    prob_3s = xr.open_dataset(
        BASEDIR / 'local/cnfaic/cnn_experiment/inference_3stage/season_v2_debris_probabilities.nc'
    )
    prob_ho = xr.open_dataset(
        BASEDIR / 'local/cnfaic/cnn_experiment/inference_human_only/season_v2_debris_probabilities.nc'
    )
    prob_2s = xr.open_dataset(
        BASEDIR / 'local/cnfaic/cnn_experiment/inference_2stage/season_v2_debris_probabilities.nc'
    )

    example_dates = ['2025-12-15', '2026-02-13', '2026-01-08']
    titles = [
        '2025-12-15 (Major event)',
        '2026-02-13 (Major event)',
        '2026-01-08 (Quiet date)',
    ]

    for date_str, title in zip(example_dates, titles):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        datasets = [
            (prob_3s, '3stage'),
            (prob_ho, 'human_only'),
            (prob_2s, '2stage'),
        ]

        for ax, (ds, name) in zip(axes, datasets):
            data = ds['debris_probability'].sel(time=date_str, method='nearest').values
            im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1, aspect='equal')
            ax.set_title(f'{name}\n(pixels>0.5: {int(np.nansum(data > 0.5)):,})')
            ax.set_xlabel('x pixel')
            ax.set_ylabel('y pixel')

        fig.suptitle(f'Debris Probability Maps: {title}', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=axes, label='Debris Probability', shrink=0.8, pad=0.02)
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])

        fname = f'example_date_{date_str}.png'
        fig.savefig(FIGDIR / fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {fname}')

    prob_3s.close()
    prob_ho.close()
    prob_2s.close()


# ============================================================
# Figure 7: Training Validation Loss
# ============================================================
def fig7_training_loss():
    logfile = BASEDIR / 'local/cnfaic/cnn_experiment/experiment_run.log'
    text = logfile.read_text()

    # Parse training stages by splitting on stage markers
    stage_markers = [
        ('Stage 1: SNFAC pretrain', 'Stage 1: Pretrain on SNFAC', 'stage1_snfac.pt'),
        ('Stage 2: CNFAIC bridge', 'Stage 2: Bridge on CNFAIC', 'stage2_bridge.pt'),
        ('Stage 3: Finetune', 'Stage 3: Finetune on CNFAIC', 'stage3_finetune.pt'),
        ('Human-only', 'Baseline A: Human-only', 'human_only.pt'),
        ('2stage: Auto pretrain', 'Baseline B: 2-stage', 'b2_auto_pretrain.pt'),
        ('2stage: Human finetune', 'Resumed from pretrained', 'b2_auto_finetune.pt'),
    ]

    # Parse all epoch lines and best model lines
    epoch_pattern = re.compile(r'epoch\s+(\d+):\s+train=([\d.]+)\s+val=([\d.]+)')
    best_pattern = re.compile(r'Saved best model \(val_loss=([\d.]+)\) at epoch (\d+)')
    complete_pattern = re.compile(r'Training complete.*saved to .*/(\S+\.pt)')

    lines = text.split('\n')

    # Group lines into training runs by finding "Training complete" boundaries
    runs = []
    current_run_epochs = []
    current_run_bests = []

    for line in lines:
        em = epoch_pattern.search(line)
        if em:
            current_run_epochs.append((int(em.group(1)), float(em.group(2)), float(em.group(3))))

        bm = best_pattern.search(line)
        if bm:
            current_run_bests.append((int(bm.group(2)), float(bm.group(1))))

        cm = complete_pattern.search(line)
        if cm:
            runs.append({
                'weight_file': cm.group(1),
                'epochs': current_run_epochs[:],
                'bests': current_run_bests[:],
            })
            current_run_epochs = []
            current_run_bests = []

    # Map weight files to display names
    name_map = {
        'stage1_snfac.pt': '3stage: S1 SNFAC pretrain',
        'stage2_bridge.pt': '3stage: S2 Bridge',
        'stage3_finetune.pt': '3stage: S3 Finetune',
        'human_only.pt': 'human_only',
        'b2_auto_pretrain.pt': '2stage: Auto pretrain',
        'b2_auto_finetune.pt': '2stage: Human finetune',
    }

    colors = {
        'stage1_snfac.pt': '#1565C0',
        'stage2_bridge.pt': '#42A5F5',
        'stage3_finetune.pt': '#90CAF9',
        'human_only.pt': '#FF9800',
        'b2_auto_pretrain.pt': '#388E3C',
        'b2_auto_finetune.pt': '#81C784',
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for run in runs:
        wf = run['weight_file']
        name = name_map.get(wf, wf)
        color = colors.get(wf, 'gray')

        # Combine epoch logs and best-model checkpoints for a complete picture
        all_points = {}
        for ep, val in run['bests']:
            all_points[ep] = val
        for ep, train, val in run['epochs']:
            all_points[ep] = val

        if all_points:
            epochs_sorted = sorted(all_points.keys())
            vals_sorted = [all_points[e] for e in epochs_sorted]
            ax.plot(epochs_sorted, vals_sorted, 'o-', label=name, color=color,
                    markersize=3, linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Validation Loss')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / 'training_val_loss.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved training_val_loss.png')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('Generating CNFAIC experiment figures...')
    print(f'Output directory: {FIGDIR}')
    print()

    fig1_model_comparison()
    fig2_dsize_detection()
    fig3_temporal_profile()
    fig456_example_dates()
    fig7_training_loss()

    print()
    print('All figures generated successfully.')
