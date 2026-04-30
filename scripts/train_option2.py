"""
train_option2.py
Fast-cWDM -- Option 2 Bidirectional MRI Synthesis
CSC 792 | University of South Dakota | Dr. Lina Chato

Architecture:  32 - 24 (condition) - 8 (target) - 1 (timestep) - embedding layer
  in_channels = 32 = 8 (target DWT) + 24 (3 condition modalities x 8 DWT each)
  Missing condition modalities are filled with zeros (8 zero channels each).

Training:
  - Iteration-based (not epoch-based): default 500k iterations
  - Full volumes (224 x 224 x 160), no patches
  - Bidirectional: random direction each iteration
  - DWT always in float32 to avoid c10::Half / float bias mismatch

Model 1: T1 <-> T1Gd
  Dir A (target=t1n): cond = [t1c, t2w, t2f]
  Dir B (target=t1c): cond = [t1n, t2w, t2f]

Model 2: T2 <-> FLAIR
  Dir A (target=t2w): cond = [t2f, t1n, t1c]
  Dir B (target=t2f): cond = [t2w, t1n, t1c]

Usage:
  python train_option2.py --pair T1_T1Gd --iters 500000
  python train_option2.py --pair T2_FLAIR --iters 500000
  python train_option2.py --pair T1_T1Gd --iters 500000 \\
      --resume /pvc/checkpoints/checkpoint_t1_t1gd_iter0100000.pt
"""

import argparse, os, sys, glob, random
import numpy as np
import nibabel
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Arg parser ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--pair',         default='T1_T1Gd',
                    choices=['T1_T1Gd', 'T2_FLAIR'])
parser.add_argument('--iters',        type=int,   default=500_000)
parser.add_argument('--lr',           type=float, default=1e-5)
parser.add_argument('--grad_accum',   type=int,   default=4)
parser.add_argument('--num_channels', type=int,   default=32)
parser.add_argument('--data_dir',
    default='/pvc/data/brats23/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
parser.add_argument('--save_dir',     default='/pvc/checkpoints')
parser.add_argument('--repo_dir',     default='/app/fast-cwdm')
parser.add_argument('--resume',       default='')
parser.add_argument('--log_every',    type=int,   default=100)
parser.add_argument('--val_every',    type=int,   default=5000)
parser.add_argument('--save_every',   type=int,   default=10000)
parser.add_argument('--num_workers',  type=int,   default=4)
args = parser.parse_args()

# ── Setup ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, args.repo_dir)
os.makedirs(args.save_dir, exist_ok=True)

from guided_diffusion.bratsloader import clip_and_normalize
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion import dist_util
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

dist_util.setup_dist(devices=[0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'PyTorch: {torch.__version__}')
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM   : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total')

# ── DWT wrappers — always float32 ─────────────────────────────────────────────
# Do NOT cast to input dtype. Casting to fp16 inside autocast causes the
# c10::Half / float bias mismatch during backward.
_dwt  = DWT_3D('haar').to(device)
_idwt = IDWT_3D('haar').to(device)

def dwt(x):
    return _dwt(x.float())

def idwt(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
    return _idwt(
        LLL.float(), LLH.float(), LHL.float(), LHH.float(),
        HLL.float(), HLH.float(), HHL.float(), HHH.float()
    )

def _dwt_8ch(vol):
    """DWT a volume to 8 channels. Returns zeros of correct shape if vol is absent."""
    if vol.abs().sum() == 0:
        B, _, H, W, D = vol.shape
        return torch.zeros(B, 8, H//2, W//2, D//2, device=device)
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(vol.to(device))
    return torch.cat([LLL/3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

# ── GradScaler compatibility ───────────────────────────────────────────────────
# torch.amp.GradScaler('cuda') was added in PyTorch 2.3.
# torch.cuda.amp.GradScaler() works from 1.6 through current.
def make_scaler():
    try:
        return torch.amp.GradScaler('cuda')
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler()

# ── Dataset ───────────────────────────────────────────────────────────────────
ALL_KEYS = ['t1n', 't1c', 't2w', 't2f']

def get_pair_samples(data_dir, key_a, key_b):
    """
    Return sample directories that have BOTH modalities of the primary pair.
    Other modalities (used as auxiliary conditions) are loaded when available
    and filled with zeros when absent.
    """
    complete = []
    for d in sorted(Path(data_dir).iterdir()):
        if not d.is_dir():
            continue
        has_a = len(glob.glob(str(d / f'*-{key_a}.nii.gz'))) > 0
        has_b = len(glob.glob(str(d / f'*-{key_b}.nii.gz'))) > 0
        if has_a and has_b:
            complete.append(str(d))
    return complete


class FullVolumeDataset(torch.utils.data.Dataset):
    """
    Loads all 4 BraTS modalities at full resolution -- no patching.
    Reference preprocessing (from bratsloader.py):
      1. clip_and_normalize: clip p0.1-p99.9, normalize to [0,1]
      2. pad z: 155 -> 160
      3. crop x,y: 240 -> 224 (remove 8px each side)
    Output per modality: (1, 224, 224, 160) float32.
    Missing modalities return zeros of the same shape.
    """
    SHAPE = (1, 224, 224, 160)

    def __init__(self, sample_dirs):
        self.dirs = sample_dirs

    def __len__(self):
        return len(self.dirs)

    def _load(self, sample_dir, key):
        paths = glob.glob(str(Path(sample_dir) / f'*-{key}.nii.gz'))
        if not paths:
            return torch.zeros(*self.SHAPE)
        try:
            vol = nibabel.load(paths[0]).get_fdata()
        except Exception as e:
            # Corrupted or truncated .nii.gz file (e.g. zlib decompression error).
            # Return zeros so compute_loss skips this sample gracefully.
            print(f'WARNING: failed to load {paths[0]}: {e}', flush=True)
            return torch.zeros(*self.SHAPE)
        vol = clip_and_normalize(vol)
        t   = torch.zeros(1, 240, 240, 160)
        t[:, :, :, :155] = torch.tensor(vol).float()
        return t[:, 8:-8, 8:-8, :].float()   # (1, 224, 224, 160)

    def __getitem__(self, idx):
        d = self.dirs[idx]
        return {k: self._load(d, k) for k in ALL_KEYS}


def infinite_loader(loader):
    """Yield batches indefinitely for iteration-based training."""
    while True:
        for batch in loader:
            yield batch


# ── Model config ──────────────────────────────────────────────────────────────
MODEL_CONFIG = dict(
    image_size           = 224,               # full volume spatial dim (no patches)
    num_channels         = args.num_channels, # 32 (reference uses 64 on A100)
    num_res_blocks       = 2,
    channel_mult         = '1,2,2,4,4',
    learn_sigma          = False,
    class_cond           = False,
    use_checkpoint       = True,              # gradient checkpointing for full volumes
    attention_resolutions= '',
    num_heads            = 1,
    num_head_channels    = -1,
    num_heads_upsample   = -1,
    use_scale_shift_norm = False,
    dropout              = 0.0,
    resblock_updown      = True,
    use_fp16             = False,
    use_new_attention_order = False,
    dims                 = 3,
    num_groups           = 32,
    in_channels          = 32,                # 8 target + 24 condition (3 x 8)
    out_channels         = 8,
    bottleneck_attention = False,
    resample_2d          = False,
    additive_skips       = False,
    use_freq             = False,
    diffusion_steps      = 1000,
    noise_schedule       = 'linear',
    timestep_respacing   = '',
    use_kl               = False,
    predict_xstart       = True,
    rescale_timesteps    = False,
    rescale_learned_sigmas = False,
    dataset              = 'brats',
    mode                 = 'i2i',
    sample_schedule      = 'direct',
)

# ── Loss ──────────────────────────────────────────────────────────────────────
def compute_loss(model, batch, diffusion, pair):
    """
    Bidirectional loss with 3 condition modalities (24 DWT channels).

    in_channels = 32 = 8 (target DWT) + 24 (3 conditions x 8 DWT)
    Missing condition modalities contribute 8 zero channels.
    pred.float() ensures mse_loss inputs are float32 even under autocast.
    """
    if pair == 'T1_T1Gd':
        dir_a = ('t1n', ['t1c', 't2w', 't2f'])
        dir_b = ('t1c', ['t1n', 't2w', 't2f'])
    else:  # T2_FLAIR
        dir_a = ('t2w', ['t2f', 't1n', 't1c'])
        dir_b = ('t2f', ['t2w', 't1n', 't1c'])

    tgt_key, cond_keys = dir_a if torch.rand(1).item() > 0.5 else dir_b

    target_vol = batch[tgt_key]
    if target_vol.numel() <= 1 or target_vol.dim() != 5:
        return None
    if target_vol.abs().sum() == 0:
        return None

    B = target_vol.shape[0]

    # DWT on target — always float32
    x0 = _dwt_8ch(target_vol)                               # (B, 8, 112, 112, 80)

    # DWT on 3 condition modalities (zeros for absent ones)
    cond = torch.cat([_dwt_8ch(batch[k]) for k in cond_keys], dim=1)  # (B, 24, ...)

    # Forward diffusion
    t        = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
    noise    = torch.randn_like(x0)
    sqrt_acp = torch.from_numpy(
        diffusion.sqrt_alphas_cumprod).float().to(device)[t][:,None,None,None,None]
    sqrt_1ma = torch.from_numpy(
        diffusion.sqrt_one_minus_alphas_cumprod).float().to(device)[t][:,None,None,None,None]
    x_t      = sqrt_acp * x0 + sqrt_1ma * noise             # (B, 8, ...)

    # UNet input: [noisy target, condition] = 32ch
    x_t_cond = torch.cat([x_t, cond], dim=1)

    pred = model(x_t_cond, t)
    return F.mse_loss(pred.float(), x0)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if args.pair == 'T1_T1Gd':
        ka, kb = 't1n', 't1c'
    else:
        ka, kb = 't2w', 't2f'

    # Dataset splits (per-pair: only the primary pair modalities are required)
    samples          = get_pair_samples(args.data_dir, ka, kb)
    train_s, valtest = train_test_split(samples, test_size=0.2, random_state=42)
    val_s, _         = train_test_split(valtest,  test_size=0.5, random_state=42)
    print(f'Pair={args.pair}  Train={len(train_s)}  Val={len(val_s)}')

    train_loader = torch.utils.data.DataLoader(
        FullVolumeDataset(train_s),
        batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        FullVolumeDataset(val_s),
        batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Build model
    model, diffusion = create_model_and_diffusion(**MODEL_CONFIG)
    model.to(device)
    diffusion.mode = 'i2i'
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters  : {total_params/1e6:.2f}M')
    print(f'in_channels : {MODEL_CONFIG["in_channels"]}  (8 target + 24 condition)')
    print(f'image_size  : {MODEL_CONFIG["image_size"]}  (full volume, no patches)')

    # Optimizer and scheduler (cosine over total optimizer steps)
    # No GradScaler: training runs in float32 (no autocast) to avoid the
    # checkpoint + autocast dtype conflict in PyTorch 2.0.0.
    num_opt_steps = args.iters // args.grad_accum
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_opt_steps, eta_min=1e-7)

    history       = {'train_iters': [], 'train_losses': [],
                     'val_iters':   [], 'val_losses':   []}
    best_val      = float('inf')
    start_iter    = 1
    running_losses = []

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        ckpt       = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        history    = ckpt.get('history', history)
        best_val   = ckpt.get('val_loss', best_val)
        start_iter = ckpt['iteration'] + 1
        # Restore scheduler state
        steps_done = (start_iter - 1) // args.grad_accum
        for _ in range(steps_done):
            scheduler.step()
        print(f'Resumed from iteration {ckpt["iteration"]}')

    print(f'\nStarting: {args.pair}')
    print(f'  iters={args.iters:,}  lr={args.lr}  grad_accum={args.grad_accum}')
    print(f'  log/{args.log_every}  val/{args.val_every}  save/{args.save_every}')
    print('-' * 65)

    train_iter = infinite_loader(train_loader)
    optimizer.zero_grad()

    for iteration in range(start_iter, args.iters + 1):
        model.train()
        batch = next(train_iter)

        # No autocast here. PyTorch 2.0.0's custom gradient checkpointing in
        # guided_diffusion/nn.py does not preserve the autocast context when it
        # re-runs the forward during backward. That causes:
        #   "Input type (HalfTensor) and weight type (FloatTensor) should be same"
        # Running fully in float32 with use_checkpoint=True is safe on V100-32GB.
        loss = compute_loss(model, batch, diffusion, args.pair)

        if loss is None:
            continue

        (loss / args.grad_accum).backward()
        running_losses.append(loss.item())

        if iteration % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if iteration % args.log_every == 0:
            avg = np.mean(running_losses[-args.log_every:])
            history['train_iters'].append(iteration)
            history['train_losses'].append(avg)
            print(f'Iter {iteration:7d}/{args.iters} | loss={avg:.4f}'
                  f' | lr={scheduler.get_last_lr()[0]:.2e}')

        if iteration % args.val_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    l = compute_loss(model, vb, diffusion, args.pair)
                    if l is not None:
                        val_losses.append(l.item())
            val_avg = np.mean(val_losses) if val_losses else 0.0
            history['val_iters'].append(iteration)
            history['val_losses'].append(val_avg)
            if val_avg < best_val:
                best_val = val_avg
                torch.save({
                    'iteration': iteration,
                    'model':     model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss':  best_val,
                    'config':    MODEL_CONFIG,
                }, f'{args.save_dir}/best_model_{args.pair.lower()}.pt')
            print(f'         VAL | val={val_avg:.4f} | best={best_val:.4f}')

        if iteration % args.save_every == 0:
            torch.save({
                'iteration': iteration,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history':   history,
                'config':    MODEL_CONFIG,
            }, f'{args.save_dir}/checkpoint_{args.pair.lower()}_iter{iteration:07d}.pt')
            print(f'  Checkpoint saved: iter {iteration}')

    print(f'\nDone. best_val={best_val:.4f}')
    print(f'Best model: {args.save_dir}/best_model_{args.pair.lower()}.pt')
