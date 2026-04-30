# Bidirectional MRI Synthesis

Bidirectional synthesis between paired MRI modalities (T1↔T1Gd, T2↔FLAIR) on BraTS23 using a wavelet-domain conditional diffusion model ([fast-cWDM](https://github.com/tsereda/fast-cwdm)).

**Course:** CSC 792 | University of South Dakota | Dr. Lina Chato

## Project Structure

```
Bidirectional-MRI-Synthesis/
├── .github/
│   └── workflows/
│       └── docker-build.yml    # Auto-builds GHCR image on push to main
├── docker/
│   ├── Dockerfile.train        # Training image (ghcr.io/amalsha27/bidirectional-mri-synthesis)
│   └── build.sh                # Manual build helper
├── nautilius/
│   ├── wandb-secret.yaml       # K8s Secret for WANDB_API_KEY
│   ├── pvc-brats23.yml         # 200Gi PersistentVolumeClaim
│   ├── pvc-uploader-pod.yaml   # Pod for uploading data to PVC
│   ├── cheatsheet.md           # Full kubectl workflow reference
│   └── jobs/
│       ├── train-t1-t1gd-full.yaml    # T1↔T1Gd, 100k iters
│       ├── train-t1-t1gd-250k.yaml   # T1↔T1Gd, resume to 250k
│       ├── train-t2-flair-full.yaml   # T2↔FLAIR, 100k iters
│       ├── train-t2-flair-250k.yaml  # T2↔FLAIR, resume to 250k
│       ├── evaluate-t1-t1gd.yaml     # Evaluate T1↔T1Gd checkpoint
│       ├── evaluate-t2-flair.yaml    # Evaluate T2↔FLAIR checkpoint
│       └── plot-loss.yaml            # Generate loss curve PNGs
├── scripts/
│   ├── train_option2.py        # Training script (fast-cWDM Option 2)
│   ├── evaluate_option2.py     # Evaluation: MSE, PSNR, SSIM + NIfTI output
│   └── plot_loss.py            # Loss curve visualisation
├── checkpoints/                # Local checkpoint cache (gitignored)
├── data/                       # Local data (gitignored)
├── results/                    # Local results (gitignored)
└── requirements.txt
```

## Models

| Model | Pair | Direction |
|-------|------|-----------|
| Model 1 | T1 ↔ T1Gd | Synthesise T1 given T1Gd+T2+FLAIR, and vice versa |
| Model 2 | T2 ↔ FLAIR | Synthesise T2 given FLAIR+T1+T1Gd, and vice versa |

Architecture: 3D UNet in wavelet (DWT) space. Input = 3 condition modalities × 8 DWT channels = 24 channels. Output = 8 DWT channels → IDWT → synthesised volume.

## Quickstart

### 1. Install dependencies (local dev)

```bash
pip install -r requirements.txt
```

### 2. One-time cluster setup

```bash
# Create PVC
kubectl apply -f nautilius/pvc-brats23.yml -n gai-lina-group

# Create WandB secret
kubectl create secret generic wandb-secret \
    --from-literal=api-key=<YOUR_WANDB_API_KEY> \
    -n gai-lina-group
```

### 3. Upload BraTS23 data to PVC

```bash
kubectl apply -f nautilius/pvc-uploader-pod.yaml -n gai-lina-group
kubectl get pod brats23-uploader -n gai-lina-group -w   # wait for Running

kubectl cp ./data/brats23/ gai-lina-group/brats23-uploader:/pvc/data/brats23/

kubectl delete pod brats23-uploader -n gai-lina-group
```

### 4. Build & push Docker image

```bash
# Option A: manual
bash docker/build.sh
docker push ghcr.io/amalsha27/bidirectional-mri-synthesis:latest

# Option B: push to main → GitHub Actions builds it automatically
git push origin main
```

### 5. Submit training jobs

```bash
# Both models in parallel (each gets its own GPU node)
kubectl apply -f nautilius/jobs/train-t1-t1gd-full.yaml -n gai-lina-group
kubectl apply -f nautilius/jobs/train-t2-flair-full.yaml -n gai-lina-group

# Follow logs
kubectl logs -f job/bidi-mri-train-t1-t1gd-full -n gai-lina-group
```

### 6. Evaluate

```bash
kubectl apply -f nautilius/jobs/evaluate-t1-t1gd.yaml -n gai-lina-group
kubectl apply -f nautilius/jobs/evaluate-t2-flair.yaml -n gai-lina-group
```

### 7. Plot loss curves

```bash
kubectl apply -f nautilius/jobs/plot-loss.yaml -n gai-lina-group
```

See `nautilius/cheatsheet.md` for the full kubectl reference (resume, debug, copy results, cleanup).

## WandB Setup

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Copy your API key from **Settings → API keys**
3. Create the K8s secret (Step 2 above)
4. All training jobs log to project **`bidirectional-mri-synthesis`** automatically

Runs appear at `https://wandb.ai/<your-username>/bidirectional-mri-synthesis`.

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| MSE | Mean Squared Error (wavelet domain, during training) | lower ↓ |
| MSE | Mean Squared Error (voxel domain, evaluation) | lower ↓ |
| PSNR | Peak Signal-to-Noise Ratio (dB) | higher ↑ |
| SSIM | Structural Similarity (slice-by-slice, middle 80%) | higher ↑ |

Evaluation is run in both synthesis directions per model (A→B and B→A).

## Results

Checkpoints and metrics land in `/pvc/` on the cluster. Copy locally:

```bash
kubectl apply -f nautilius/pvc-uploader-pod.yaml -n gai-lina-group

kubectl cp gai-lina-group/brats23-uploader:/pvc/checkpoints/ ./checkpoints/
kubectl cp gai-lina-group/brats23-uploader:/pvc/results/ ./results/

kubectl delete pod brats23-uploader -n gai-lina-group
```

## PVC Layout (on cluster)

```
/pvc/
├── data/brats23/
│   ├── train/        # 80% split
│   └── train_full/   # full dataset
├── checkpoints/
│   ├── t1_t1gd_full/
│   │   ├── best_model_t1_t1gd.pt
│   │   └── checkpoint_t1_t1gd_iter0XXXXXX.pt
│   └── t2_flair_full/
│       ├── best_model_t2_flair.pt
│       └── checkpoint_t2_flair_iter0XXXXXX.pt
└── results/
    ├── metrics_t1_t1gd_dirA.csv
    ├── metrics_t2_flair_dirB.csv
    ├── summary_*.txt
    ├── loss_curves_*.png
    └── slices_*/
```
