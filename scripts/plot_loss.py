"""
plot_loss.py
Fast-cWDM Option 2 -- Loss Curve Generator
CSC 792 | University of South Dakota | Dr. Lina Chato

Loads one or two checkpoints and produces a loss curve PNG.
No GPU required -- checkpoints are loaded on CPU.

Usage (on cluster, pointing at PVC paths):
  python plot_loss.py \
      --t1_ckpt   /pvc/checkpoints/checkpoint_t1_t1gd_iter0100000.pt \
      --t2_ckpt   /pvc/checkpoints/checkpoint_t2_flair_iter0100000.pt \
      --save_dir  /pvc/results

Usage (locally, after downloading checkpoints):
  python plot_loss.py \
      --t1_ckpt   checkpoint_t1_t1gd_iter0100000.pt \
      --t2_ckpt   checkpoint_t2_flair_iter0100000.pt \
      --save_dir  .
"""

import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--t1_ckpt',  default='', help='Path to T1_T1Gd checkpoint (.pt)')
parser.add_argument('--t2_ckpt',  default='', help='Path to T2_FLAIR checkpoint (.pt)')
parser.add_argument('--save_dir', default='/pvc/results')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

def load_history(ckpt_path):
    """Load checkpoint on CPU and return (history_dict, iteration)."""
    print(f'Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    iteration = ckpt.get('iteration', ckpt.get('epoch', '?'))
    history   = ckpt.get('history', {})
    print(f'  Checkpoint iteration : {iteration}')
    print(f'  Train loss points    : {len(history.get("train_iters", []))}')
    print(f'  Val loss points      : {len(history.get("val_iters", []))}')
    return history, iteration

def smooth(values, window=10):
    """Simple moving average for readability."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

# ── Collect what is available ─────────────────────────────────────────────────
datasets = []

if args.t1_ckpt and os.path.exists(args.t1_ckpt):
    h, it = load_history(args.t1_ckpt)
    datasets.append(('T1 <-> T1Gd (Model 1)', h, it, '#2196F3', '#FF9800'))
else:
    if args.t1_ckpt:
        print(f'WARNING: T1_T1Gd checkpoint not found at {args.t1_ckpt}')

if args.t2_ckpt and os.path.exists(args.t2_ckpt):
    h, it = load_history(args.t2_ckpt)
    datasets.append(('T2 <-> FLAIR (Model 2)', h, it, '#4CAF50', '#E91E63'))
else:
    if args.t2_ckpt:
        print(f'WARNING: T2_FLAIR checkpoint not found at {args.t2_ckpt}')

if not datasets:
    print('No checkpoints found. Provide at least one of --t1_ckpt or --t2_ckpt.')
    raise SystemExit(1)

# ── Figure layout: one row per model ─────────────────────────────────────────
n_models = len(datasets)
fig, axes = plt.subplots(n_models, 1, figsize=(11, 4 * n_models), squeeze=False)
fig.suptitle('Fast-cWDM Option 2 — Training and Validation Loss',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (label, history, iteration, train_color, val_color) in zip(axes[:, 0], datasets):
    ti = history.get('train_iters',  [])
    tl = history.get('train_losses', [])
    vi = history.get('val_iters',    [])
    vl = history.get('val_losses',   [])

    if ti and tl:
        # Raw training loss (thin, low opacity)
        ax.plot(ti, tl, color=train_color, alpha=0.25, linewidth=0.7,
                label='_nolegend_')
        # Smoothed training loss
        if len(tl) >= 10:
            smooth_x = ti[9:]    # convolve shortens by window-1
            smooth_y = smooth(tl, window=10)
            ax.plot(smooth_x, smooth_y, color=train_color, linewidth=2.0,
                    label=f'Train loss (smoothed, window=10)')
        else:
            ax.plot(ti, tl, color=train_color, linewidth=2.0, label='Train loss')

    if vi and vl:
        ax.plot(vi, vl, color=val_color, linewidth=2.0,
                marker='o', markersize=4, label='Validation loss')
        # Mark best val
        best_idx = int(np.argmin(vl))
        ax.scatter(vi[best_idx], vl[best_idx], color=val_color,
                   s=80, zorder=5, label=f'Best val = {vl[best_idx]:.5f} @ iter {vi[best_idx]:,}')

    ax.set_title(f'{label}  (checkpoint @ iteration {iteration:,})'
                 if isinstance(iteration, int) else f'{label}', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('Loss (MSE, wavelet domain)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate final values
    if tl:
        ax.annotate(f'Final train: {tl[-1]:.5f}',
                    xy=(ti[-1], tl[-1]),
                    xytext=(-80, 12), textcoords='offset points',
                    fontsize=8, color=train_color,
                    arrowprops=dict(arrowstyle='->', color=train_color, lw=1))
    if vl:
        ax.annotate(f'Final val: {vl[-1]:.5f}',
                    xy=(vi[-1], vl[-1]),
                    xytext=(-80, -20), textcoords='offset points',
                    fontsize=8, color=val_color,
                    arrowprops=dict(arrowstyle='->', color=val_color, lw=1))

plt.tight_layout()

# Save
iter_tag = '_'.join(
    str(d[2]) for d in datasets if isinstance(d[2], int)
) or 'combined'
out_path = os.path.join(args.save_dir, f'loss_curves_{iter_tag}.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nPlot saved: {out_path}')

# ── Also print a quick summary table ─────────────────────────────────────────
print('\n--- Loss Summary ---')
for label, history, iteration, _, _ in datasets:
    tl = history.get('train_losses', [])
    vl = history.get('val_losses',   [])
    print(f'\n{label}  (iter {iteration}):')
    if tl:
        print(f'  Train loss -- first: {tl[0]:.5f}  |  last: {tl[-1]:.5f}  '
              f'|  min: {min(tl):.5f}')
    if vl:
        print(f'  Val loss   -- first: {vl[0]:.5f}  |  last: {vl[-1]:.5f}  '
              f'|  best: {min(vl):.5f} @ iter {history["val_iters"][int(np.argmin(vl))]:,}')
