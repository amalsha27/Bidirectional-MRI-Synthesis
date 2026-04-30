"""
BraTS 2023 — Full EDA Script
Fast-cWDM Option 2 | CSC 792 | University of South Dakota | Dr. Lina Chato

Saves all outputs to --save_dir (default: /pvc/results/EDA)

Sections:
  1. Dataset completeness & structure
  2. Volume shape & voxel spacing
  3. Intensity statistics (foreground voxels)
  4. Intensity histograms (per modality)
  5. Sample axial slices (grid)
  6. Three-plane view (axial / coronal / sagittal)
  7. Before / after normalization
  8. Modality pair correlations (T1↔T1Gd, T2↔FLAIR)
  9. 3D Haar DWT sub-band visualisation
 10. Train / Val / Test split breakdown
 11. Summary report (TXT + CSV)
"""

import os, sys, json, csv, argparse, warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pywt

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',  default='/pvc/data/brats23/train')
parser.add_argument('--save_dir',  default='/pvc/results/EDA')
parser.add_argument('--n_stats',   type=int, default=100,
                    help='Subjects used for intensity stats (0 = all)')
parser.add_argument('--n_viz',     type=int, default=6,
                    help='Subjects shown in slice grid')
parser.add_argument('--n_corr',    type=int, default=20,
                    help='Subjects used for correlation plots')
args = parser.parse_args()

SAVE_DIR  = Path(args.save_dir)
DATA_DIR  = Path(args.data_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODALITIES = {'t1n': 'T1', 't1c': 'T1Gd', 't2w': 'T2', 't2f': 'FLAIR'}
MOD_COLORS = {'T1': '#4A90E2', 'T1Gd': '#E2784A', 'T2': '#4AE27A', 'FLAIR': '#E2E24A'}
TARGET     = (224, 224, 160)
BG         = '#0f0f0f'
AX_BG      = '#1a1a1a'

print(f"\n{'='*65}")
print(f"  BraTS 2023 Full EDA")
print(f"  Data : {DATA_DIR}")
print(f"  Out  : {SAVE_DIR}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_mod_path(folder, key):
    hits = list(folder.glob(f'*-{key}.nii.gz'))
    return hits[0] if hits else None

def load_vol(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)

def disp(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-8)

def crop_pad(vol, tgt=TARGET):
    out = np.zeros(tgt, dtype=np.float32)
    sx, sy, sz = vol.shape
    tx, ty, tz = tgt
    cx, cy, cz = min(sx,tx), min(sy,ty), min(sz,tz)
    xs=(sx-cx)//2; ys=(sy-cy)//2; zs=(sz-cz)//2
    xt=(tx-cx)//2; yt=(ty-cy)//2; zt=(tz-cz)//2
    out[xt:xt+cx, yt:yt+cy, zt:zt+cz] = vol[xs:xs+cx, ys:ys+cy, zs:zs+cz]
    return out

def normalize(vol):
    out = np.zeros_like(vol)
    fg  = vol > 0
    if not fg.any(): return out
    v   = vol[fg]
    lo, hi = v.min(), v.max()
    out[fg] = (v - lo) / (hi - lo + 1e-8)
    return out

def dwt3d(vol):
    """3D Haar DWT → 8 sub-bands, LLL scaled by 1/3."""
    coeffs = pywt.dwtn(vol, 'haar')
    order  = ['aaa','aad','ada','add','daa','dad','dda','ddd']
    bands  = np.stack([coeffs[k] for k in order], axis=0).astype(np.float32)
    bands[0] /= 3.0
    return bands

def savefig(name):
    path = SAVE_DIR / name
    plt.savefig(str(path), dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close('all')
    print(f"  Saved → {name}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPLETENESS CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("[ 1/11 ] Scanning dataset completeness …")

all_folders = sorted([d for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name.startswith('BraTS')])

complete, incomplete, miss_count = [], [], {}
for d in tqdm(all_folders, desc='  checking', ncols=70):
    present = [k for k in MODALITIES if get_mod_path(d, k) is not None]
    missing = [k for k in MODALITIES if k not in present]
    if len(missing) == 0:
        complete.append(d)
    else:
        incomplete.append(d)
        for m in missing:
            miss_count[m] = miss_count.get(m, 0) + 1

print(f"\n  Total folders  : {len(all_folders)}")
print(f"  Complete (4/4) : {len(complete)}")
print(f"  Incomplete     : {len(incomplete)}")
print(f"  Missing counts : {miss_count}\n")

# Bar chart
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(AX_BG)
bars = ax.bar(['Complete\n(all 4 mod)', 'Incomplete\n(missing mod)'],
              [len(complete), len(incomplete)],
              color=['#2ECC71', '#E74C3C'], width=0.5)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
            str(int(b.get_height())), ha='center', color='white', fontsize=13)
if miss_count:
    miss_labels = [f"missing\n{k}: {v}" for k, v in miss_count.items()]
    ax.set_xlabel('  '.join(miss_labels), color='#aaa', fontsize=9)
ax.set_title('BraTS 2023 GLI — Dataset Completeness', color='white', fontsize=13)
ax.tick_params(colors='gray'); ax.spines['bottom'].set_color('#444')
ax.spines['left'].set_color('#444'); ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.label.set_color('gray')
savefig('01_completeness.png')

samples = complete          # work only with complete samples from here on
N = len(samples)

# ─────────────────────────────────────────────────────────────────────────────
# 2. VOLUME SHAPE & SPACING
# ─────────────────────────────────────────────────────────────────────────────
print("[ 2/11 ] Checking shapes & voxel spacing …")

n_check = min(50, N)
shapes, spacings = [], []
for d in tqdm(samples[:n_check], desc='  reading headers', ncols=70):
    img  = nib.load(str(get_mod_path(d, 't1n')))
    shapes.append(img.shape)
    spacings.append(tuple(round(float(z), 3) for z in img.header.get_zooms()[:3]))

uniq_sh = {}
for s in shapes:
    uniq_sh[s] = uniq_sh.get(s, 0) + 1
uniq_sp = {}
for s in spacings:
    uniq_sp[s] = uniq_sp.get(s, 0) + 1

print(f"  Unique shapes   : {uniq_sh}")
print(f"  Unique spacings : {uniq_sp}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. INTENSITY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
print("[ 3/11 ] Computing intensity statistics …")

n_stats = N if args.n_stats <= 0 else min(args.n_stats, N)
all_stats = {label: {'mean':[], 'std':[], 'min':[], 'max':[], 'p01':[], 'p99':[]}
             for label in MODALITIES.values()}

for d in tqdm(samples[:n_stats], desc='  stats', ncols=70):
    for key, label in MODALITIES.items():
        vol = load_vol(get_mod_path(d, key))
        fg  = vol[vol > 0]
        if len(fg) == 0: continue
        all_stats[label]['mean'].append(fg.mean())
        all_stats[label]['std'].append(fg.std())
        all_stats[label]['min'].append(fg.min())
        all_stats[label]['max'].append(fg.max())
        all_stats[label]['p01'].append(np.percentile(fg, 1))
        all_stats[label]['p99'].append(np.percentile(fg, 99))

agg = {}
for lbl, d in all_stats.items():
    agg[lbl] = {k: float(np.mean(v)) for k, v in d.items()}

print(f"\n  {'Modality':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>10} {'p99':>8}")
print(f"  {'-'*52}")
for lbl, s in agg.items():
    print(f"  {lbl:<8} {s['mean']:>8.1f} {s['std']:>8.1f} "
          f"{s['min']:>8.1f} {s['max']:>10.1f} {s['p99']:>8.1f}")
print()

# Save stats CSV
with open(SAVE_DIR / 'stats_intensity.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['modality','mean','std','min','max','p01','p99'])
    for lbl, s in agg.items():
        w.writerow([lbl, f"{s['mean']:.2f}", f"{s['std']:.2f}",
                    f"{s['min']:.2f}", f"{s['max']:.2f}",
                    f"{s['p01']:.2f}", f"{s['p99']:.2f}"])
print("  Saved → stats_intensity.csv")

# Box plot
fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor=BG)
fig.suptitle(f'Intensity Distribution — Foreground Voxels ({n_stats} subjects)',
             color='white', fontsize=13)
for ax, (lbl, st) in zip(axes, all_stats.items()):
    ax.set_facecolor(AX_BG)
    bp = ax.boxplot(st['mean'], patch_artist=True,
                    boxprops=dict(facecolor=MOD_COLORS[lbl], alpha=0.7),
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='gray'),
                    capprops=dict(color='gray'),
                    flierprops=dict(marker='.', color='gray', markersize=3))
    ax.bar([1], [np.mean(st['mean'])], width=0.5,
           color=MOD_COLORS[lbl], alpha=0.6, label='mean')
    ax.set_title(lbl, color='white', fontsize=12)
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_xticks([])
    ax.text(1, np.mean(st['mean']), f"μ={np.mean(st['mean']):.0f}",
            ha='center', va='bottom', color='white', fontsize=10)
plt.tight_layout()
savefig('02_intensity_boxplot.png')

# ─────────────────────────────────────────────────────────────────────────────
# 4. INTENSITY HISTOGRAMS
# ─────────────────────────────────────────────────────────────────────────────
print("[ 4/11 ] Generating intensity histograms …")

n_hist = min(50, N)
all_vox = {lbl: [] for lbl in MODALITIES.values()}

for d in tqdm(samples[:n_hist], desc='  histograms', ncols=70):
    for key, lbl in MODALITIES.items():
        vol = load_vol(get_mod_path(d, key))
        fg  = vol[vol > 0].flatten()
        idx = np.random.choice(len(fg), min(5000, len(fg)), replace=False)
        all_vox[lbl].extend(fg[idx].tolist())

fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
fig.suptitle(f'Foreground Intensity Histograms ({n_hist} subjects)',
             color='white', fontsize=13)
axes = axes.flatten()
for i, (lbl, vox) in enumerate(all_vox.items()):
    ax = axes[i]
    ax.set_facecolor(AX_BG)
    v = np.array(vox)
    ax.hist(v, bins=150, color=MOD_COLORS[lbl], alpha=0.85, density=True)
    p01 = np.percentile(v, 1);  p99 = np.percentile(v, 99)
    ax.axvline(v.mean(), color='white', lw=1.5, linestyle='--',
               label=f'mean={v.mean():.0f}')
    ax.axvline(p01, color='cyan', lw=1, linestyle=':', label=f'p1={p01:.0f}')
    ax.axvline(p99, color='red',  lw=1, linestyle=':', label=f'p99={p99:.0f}')
    ax.set_title(lbl, color='white', fontsize=12)
    ax.set_xlabel('Intensity', color='gray', fontsize=9)
    ax.set_ylabel('Density',   color='gray', fontsize=9)
    ax.tick_params(colors='gray')
    ax.legend(fontsize=9, labelcolor='white', facecolor='#2a2a2a', framealpha=0.8)
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
plt.tight_layout()
savefig('03_histograms.png')

# ─────────────────────────────────────────────────────────────────────────────
# 5. SAMPLE AXIAL SLICES GRID
# ─────────────────────────────────────────────────────────────────────────────
print("[ 5/11 ] Plotting sample axial slices …")

n_viz = min(args.n_viz, N)
fig, axes = plt.subplots(n_viz, 4, figsize=(16, 4 * n_viz), facecolor=BG)
if n_viz == 1: axes = axes[np.newaxis, :]
fig.suptitle(f'BraTS 2023 GLI — Middle Axial Slices (first {n_viz} complete subjects)',
             color='white', fontsize=13, y=1.01)

for row, d in enumerate(samples[:n_viz]):
    for col, (key, lbl) in enumerate(MODALITIES.items()):
        ax = axes[row, col]
        ax.set_facecolor(AX_BG)
        vol = load_vol(get_mod_path(d, key))
        mid = vol.shape[2] // 2
        ax.imshow(np.rot90(disp(vol[:, :, mid])), cmap='gray', aspect='equal')
        if row == 0: ax.set_title(lbl, color='white', fontsize=12, pad=6)
        if col == 0: ax.set_ylabel(d.name[-14:], color='gray', fontsize=8)
        ax.axis('off')

plt.tight_layout()
savefig('04_axial_slices.png')

# ─────────────────────────────────────────────────────────────────────────────
# 6. THREE-PLANE VIEW
# ─────────────────────────────────────────────────────────────────────────────
print("[ 6/11 ] Three-plane view …")

d   = samples[0]
fig, axes = plt.subplots(4, 3, figsize=(14, 18), facecolor=BG)
fig.suptitle(f'Three-Plane View — {d.name}', color='white', fontsize=12)

for row, (key, lbl) in enumerate(MODALITIES.items()):
    vol = load_vol(get_mod_path(d, key))
    slices = [
        np.rot90(disp(vol[:, :, vol.shape[2] // 2])),   # axial
        np.rot90(disp(vol[:, vol.shape[1] // 2, :])),   # coronal
        np.rot90(disp(vol[vol.shape[0] // 2, :, :])),   # sagittal
    ]
    titles = ['Axial', 'Coronal', 'Sagittal']
    for col, (slc, t) in enumerate(zip(slices, titles)):
        ax = axes[row, col]
        ax.set_facecolor(AX_BG)
        ax.imshow(slc, cmap='gray')
        if row == 0: ax.set_title(t, color='white', fontsize=11)
        if col == 0: ax.set_ylabel(lbl, color=MOD_COLORS[lbl], fontsize=11)
        ax.axis('off')

plt.tight_layout()
savefig('05_three_planes.png')

# ─────────────────────────────────────────────────────────────────────────────
# 7. BEFORE / AFTER NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("[ 7/11 ] Before/after normalisation …")

d   = samples[0]
raw = {}; proc = {}
for key, lbl in MODALITIES.items():
    v          = load_vol(get_mod_path(d, key))
    raw[lbl]   = v
    proc[lbl]  = normalize(crop_pad(v))

fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=BG)
fig.suptitle(f'Before vs After Preprocessing — {d.name}', color='white', fontsize=12)
for col, (key, lbl) in enumerate(MODALITIES.items()):
    for ri, (vols, tag) in enumerate([(raw, 'Raw'), (proc, 'Normalized [0,1]')]):
        ax = axes[ri, col]
        ax.set_facecolor(AX_BG)
        v   = vols[lbl]
        mid = v.shape[2] // 2
        slc = disp(v[:, :, mid]) if ri == 0 else v[:, :, mid]
        ax.imshow(np.rot90(slc), cmap='gray', vmin=0, vmax=1)
        if ri == 0:
            fg = raw[lbl][raw[lbl] > 0]
            ax.set_title(f'{lbl}\n[{fg.min():.0f}, {fg.max():.0f}]',
                         color='white', fontsize=10)
        else:
            fg = proc[lbl][proc[lbl] > 0]
            ax.set_title(f'mean={fg.mean():.4f}', color='white', fontsize=10)
        if col == 0: ax.set_ylabel(tag, color='gray', fontsize=10)
        ax.axis('off')

plt.tight_layout()
savefig('06_before_after_norm.png')

# ─────────────────────────────────────────────────────────────────────────────
# 8. MODALITY PAIR CORRELATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("[ 8/11 ] Modality pair correlations …")

n_corr = min(args.n_corr, N)
pairs  = [('t1n','t1c','T1','T1Gd','#4A90E2'),
          ('t2w','t2f','T2','FLAIR','#4AE27A')]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle(f'Voxel Correlation — Synthesis Pairs ({n_corr} subjects)',
             color='white', fontsize=13)
pearson_r = {}

for ax, (k1, k2, l1, l2, col) in zip(axes, pairs):
    ax.set_facecolor(AX_BG)
    xs, ys = [], []
    for d in tqdm(samples[:n_corr], desc=f'  {l1}↔{l2}', ncols=70):
        v1 = load_vol(get_mod_path(d, k1))
        v2 = load_vol(get_mod_path(d, k2))
        mask = (v1 > 0) & (v2 > 0)
        fg1  = v1[mask].flatten(); fg2 = v2[mask].flatten()
        idx  = np.random.choice(len(fg1), min(3000, len(fg1)), replace=False)
        xs.extend(fg1[idx].tolist()); ys.extend(fg2[idx].tolist())
    r = float(np.corrcoef(xs, ys)[0, 1])
    pearson_r[f'{l1}_{l2}'] = round(r, 4)
    ax.scatter(xs, ys, alpha=0.08, s=1.5, color=col)
    ax.set_xlabel(l1, color='gray', fontsize=10)
    ax.set_ylabel(l2, color='gray', fontsize=10)
    ax.set_title(f'{l1} ↔ {l2}', color='white', fontsize=12)
    ax.tick_params(colors='gray')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.text(0.05, 0.93, f'Pearson r = {r:.4f}',
            transform=ax.transAxes, color='white', fontsize=11,
            bbox=dict(boxstyle='round', fc='#222', alpha=0.85))
    print(f"  {l1} ↔ {l2}  →  r = {r:.4f}")

plt.tight_layout()
savefig('07_correlations.png')

# ─────────────────────────────────────────────────────────────────────────────
# 9. DWT SUB-BAND VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("[ 9/11 ] DWT sub-band visualisation …")

d   = samples[0]
vol = normalize(crop_pad(load_vol(get_mod_path(d, 't1n'))))
bands = dwt3d(vol)          # (8, 112, 112, 80)
band_names = ['LLL÷3','LLH','LHL','LHH','HLL','HLH','HHL','HHH']
mid = bands.shape[3] // 2

fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=BG)
fig.suptitle(f'3D Haar DWT Sub-bands — T1  (axial slice)  {d.name}',
             color='white', fontsize=12)
band_cols = ['#00B4D8','#2ECC71','#2ECC71','#2ECC71',
             '#F39614','#F39614','#F39614','#E74C3C']
axes = axes.flatten()
for i, (name, col) in enumerate(zip(band_names, band_cols)):
    ax = axes[i]
    ax.set_facecolor(AX_BG)
    slc = disp(np.rot90(bands[i, :, :, mid]))
    ax.imshow(slc, cmap='gray')
    ax.set_title(f'ch {i}: {name}', color=col, fontsize=11)
    ax.axis('off')

plt.tight_layout()
savefig('08_dwt_subbands.png')

# Sub-band energy bar chart
energies = [float(np.mean(bands[i]**2)) for i in range(8)]
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(AX_BG)
bars = ax.bar(band_names, energies,
              color=['#00B4D8']+['#2ECC71']*3+['#F39614']*3+['#E74C3C'])
ax.set_title('DWT Sub-band Mean Squared Energy (T1 volume)', color='white', fontsize=12)
ax.set_xlabel('Sub-band', color='gray'); ax.set_ylabel('Mean energy', color='gray')
ax.tick_params(colors='gray')
for sp in ax.spines.values(): sp.set_edgecolor('#444')
for b, e in zip(bars, energies):
    ax.text(b.get_x()+b.get_width()/2, b.get_height(),
            f'{e:.4f}', ha='center', va='bottom', color='white', fontsize=9)
plt.tight_layout()
savefig('08b_dwt_energy.png')

# ─────────────────────────────────────────────────────────────────────────────
# 10. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("[10/11 ] Train / Val / Test split …")

train_s, valtest = train_test_split(samples, test_size=0.20, random_state=42)
val_s,   test_s  = train_test_split(valtest, test_size=0.50, random_state=42)

print(f"\n  Total complete : {N}")
print(f"  Train          : {len(train_s)}   ({len(train_s)/N*100:.1f}%)")
print(f"  Val            : {len(val_s)}    ({len(val_s)/N*100:.1f}%)")
print(f"  Test           : {len(test_s)}    ({len(test_s)/N*100:.1f}%)\n")

# Save split lists as JSON
split_info = {
    'total_complete': N,
    'train': [d.name for d in train_s],
    'val':   [d.name for d in val_s],
    'test':  [d.name for d in test_s],
}
with open(SAVE_DIR / 'split_subjects.json', 'w') as f:
    json.dump(split_info, f, indent=2)
print("  Saved → split_subjects.json")

# Pie chart
fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
ax.set_facecolor(BG)
sizes  = [len(train_s), len(val_s), len(test_s)]
labels = [f'Train\n{len(train_s)} subjects\n({len(train_s)/N*100:.1f}%)',
          f'Val\n{len(val_s)} subjects\n({len(val_s)/N*100:.1f}%)',
          f'Test\n{len(test_s)} subjects\n({len(test_s)/N*100:.1f}%)']
colors = ['#2ECC71', '#F39614', '#9B59B6']
wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                       startangle=140, textprops=dict(color='white', fontsize=11))
ax.set_title('Train / Val / Test Split\n(random_state=42)',
             color='white', fontsize=13, pad=20)
plt.tight_layout()
savefig('09_split_pie.png')

# ─────────────────────────────────────────────────────────────────────────────
# 11. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("[11/11 ] Writing summary report …")

lines = [
    "=" * 65,
    "  BraTS 2023 GLI — EDA Summary Report",
    "  Fast-cWDM Option 2 | CSC 792 | Univ. of South Dakota",
    "=" * 65,
    "",
    f"Data directory  : {DATA_DIR}",
    f"Save directory  : {SAVE_DIR}",
    "",
    "── COMPLETENESS ──────────────────────────────────────────────",
    f"  Total folders    : {len(all_folders)}",
    f"  Complete (4/4)   : {len(complete)}",
    f"  Incomplete       : {len(incomplete)}",
    f"  Missing modality : {miss_count}",
    "",
    "── VOLUME SHAPE & SPACING ────────────────────────────────────",
    f"  Unique shapes    : {uniq_sh}",
    f"  Unique spacings  : {uniq_sp}",
    f"  Preprocessing    : (240,240,155) → center-crop → (224,224,160)",
    "",
    "── INTENSITY STATISTICS (foreground, {n_stats} subjects) ──────────────".format(n_stats=n_stats),
    f"  {'Modality':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>10} {'p99':>8}",
    "  " + "-"*52,
]
for lbl, s in agg.items():
    lines.append(f"  {lbl:<8} {s['mean']:>8.1f} {s['std']:>8.1f} "
                 f"{s['min']:>8.1f} {s['max']:>10.1f} {s['p99']:>8.1f}")

lines += [
    "",
    "── MODALITY PAIR CORRELATIONS ────────────────────────────────",
]
for pair, r in pearson_r.items():
    lines.append(f"  {pair:<20} Pearson r = {r:.4f}")

lines += [
    "",
    "── TRAIN / VAL / TEST SPLIT (random_state=42) ────────────────",
    f"  Train : {len(train_s):4d} subjects  ({len(train_s)*2} training examples/model)",
    f"  Val   : {len(val_s):4d} subjects  ({len(val_s)*2} validation examples/model)",
    f"  Test  : {len(test_s):4d} subjects  ({len(test_s)*2} test examples/model)",
    "",
    "── OUTPUT FILES ──────────────────────────────────────────────",
    "  01_completeness.png",
    "  02_intensity_boxplot.png",
    "  03_histograms.png",
    "  04_axial_slices.png",
    "  05_three_planes.png",
    "  06_before_after_norm.png",
    "  07_correlations.png",
    "  08_dwt_subbands.png",
    "  08b_dwt_energy.png",
    "  09_split_pie.png",
    "  stats_intensity.csv",
    "  split_subjects.json",
    "  summary_eda.txt",
    "",
    "=" * 65,
]

report = "\n".join(lines)
print(report)

with open(SAVE_DIR / 'summary_eda.txt', 'w') as f:
    f.write(report)
print("\n  Saved → summary_eda.txt")

print(f"\n✓ EDA complete — all outputs in {SAVE_DIR}\n")
