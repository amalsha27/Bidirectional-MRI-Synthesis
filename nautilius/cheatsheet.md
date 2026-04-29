# Nautilus NRP — kubectl Cheatsheet for Bidirectional MRI Synthesis

## One-time setup

```bash
# 1. Create PVC (200Gi shared storage)
kubectl apply -f nautilius/pvc-brats23.yml -n gai-lina-group
kubectl get pvc brats23-pvc -n gai-lina-group -w

# 2. Create WandB secret (recommended: avoids key in YAML)
kubectl create secret generic wandb-secret \
    --from-literal=api-key=<YOUR_WANDB_API_KEY> \
    -n gai-lina-group

# 3. Verify secret
kubectl get secret wandb-secret -n gai-lina-group
```

## Build & push Docker image

```bash
# Option A: Manual (local Docker)
bash docker/build.sh
docker push ghcr.io/kagozi/bidirectional-mri-synthesis:latest

# Option B: GitHub Actions (auto on push to main)
git push origin main   # triggers .github/workflows/docker-build.yml

# Login to GHCR first (one-time):
echo $GITHUB_TOKEN | docker login ghcr.io -u kagozi --password-stdin
```

## Upload data to PVC

```bash
# Start uploader pod
kubectl apply -f nautilius/pvc-uploader-pod.yaml -n gai-lina-group
kubectl get pod brats23-uploader -n gai-lina-group -w   # wait for Running

# Copy BraTS23 data
kubectl cp ./data/brats23/ gai-lina-group/brats23-uploader:/pvc/data/brats23/

# Or shell in to extract archives
kubectl exec -it brats23-uploader -n gai-lina-group -- bash

# Delete uploader pod when done
kubectl delete pod brats23-uploader -n gai-lina-group
```

## Run training jobs

```bash
# T1 <-> T1Gd (full dataset, 100k iters)
kubectl apply -f nautilius/jobs/train-t1-t1gd-full.yaml -n gai-lina-group
kubectl logs -f job/bidi-mri-train-t1-t1gd-full -n gai-lina-group

# T2 <-> FLAIR (full dataset, 100k iters)
kubectl apply -f nautilius/jobs/train-t2-flair-full.yaml -n gai-lina-group
kubectl logs -f job/bidi-mri-train-t2-flair-full -n gai-lina-group

# Resume runs (250k total iters)
kubectl apply -f nautilius/jobs/train-t1-t1gd-250k.yaml -n gai-lina-group
kubectl apply -f nautilius/jobs/train-t2-flair-250k.yaml -n gai-lina-group

# Monitor all jobs
kubectl get jobs -n gai-lina-group -w
```

## Run evaluation jobs

```bash
kubectl apply -f nautilius/jobs/evaluate-t1-t1gd.yaml -n gai-lina-group
kubectl apply -f nautilius/jobs/evaluate-t2-flair.yaml -n gai-lina-group
kubectl logs -f job/bidi-mri-eval-t1-t1gd -n gai-lina-group
```

## Plot loss curves

```bash
kubectl apply -f nautilius/jobs/plot-loss.yaml -n gai-lina-group
kubectl logs -f job/bidi-mri-plot-loss -n gai-lina-group
```

## Monitor & debug

```bash
# Check job status
kubectl describe job bidi-mri-train-t1-t1gd-full -n gai-lina-group

# Follow logs
kubectl logs -f job/bidi-mri-train-t1-t1gd-full -n gai-lina-group

# Shell into a running job pod
kubectl exec -it \
  $(kubectl get pods -l job-name=bidi-mri-train-t1-t1gd-full -n gai-lina-group \
    -o jsonpath='{.items[0].metadata.name}') \
  -n gai-lina-group -- bash

# GPU check inside pod
nvidia-smi

# Check PVC contents
kubectl exec brats23-uploader -n gai-lina-group -- ls -lah /pvc/checkpoints/
```

## Cleanup

```bash
# Delete a completed job
kubectl delete job bidi-mri-train-t1-t1gd-full -n gai-lina-group

# Delete all bidirectional-mri jobs
kubectl delete jobs -l app=bidirectional-mri-synthesis -n gai-lina-group

# Delete PVC (WARNING: destroys all data)
kubectl delete pvc brats23-pvc -n gai-lina-group
```

## Copy results from PVC

```bash
# Restart uploader pod first
kubectl apply -f nautilius/pvc-uploader-pod.yaml -n gai-lina-group
kubectl get pod brats23-uploader -n gai-lina-group -w

# Copy checkpoints
kubectl cp gai-lina-group/brats23-uploader:/pvc/checkpoints/t1_t1gd_full/best_model_t1_t1gd.pt \
    ./checkpoints/best_model_t1_t1gd.pt

# Copy results / plots
kubectl cp gai-lina-group/brats23-uploader:/pvc/results/ ./results/
```

## PVC directory layout

```
/pvc/
├── data/
│   └── brats23/
│       ├── train/        # 80% split used for 250k runs
│       └── train_full/   # full dataset
├── checkpoints/
│   ├── t1_t1gd_full/
│   │   ├── best_model_t1_t1gd.pt
│   │   └── checkpoint_t1_t1gd_iter0XXXXXX.pt
│   ├── t1_t1gd_250k/
│   ├── t2_flair_full/
│   └── t2_flair_250k/
└── results/
    ├── metrics_*.csv
    ├── summary_*.txt
    ├── loss_curves_*.png
    ├── slices_t1_t1gd/
    └── slices_t2_flair/
```
