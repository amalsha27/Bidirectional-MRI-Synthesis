# Kubernetes Job Submission Guide
# Fast-cWDM Option 2 — NRP Nautilus Cluster
# CSC 792 | University of South Dakota | Dr. Lina Chato

---

## Prerequisites

- `kubectl` installed on your local machine
- Access to the NRP Nautilus cluster (kubeconfig configured)
- Namespace: `gai-lina-group`
- PVC: `brats23-pvc`

---

## File Overview

```
k8s/
  train_option2.py        # Training script (iteration-based, full volumes, 32ch)
  script-uploader.yaml    # Utility pod for copying script to PVC
  train-t1-t1gd.yaml      # Job: Model 1 (T1 <-> T1Gd), 500k iterations
  train-t2-flair.yaml     # Job: Model 2 (T2 <-> FLAIR), 500k iterations
```

---

## Step 1 — Set kubectl context to Nautilus

Check your current context:
```bash
kubectl config current-context
```

If it is not pointing to Nautilus, set it:
```bash
kubectl config use-context nautilus
```

Verify you can reach the cluster:
```bash
kubectl get nodes -n gai-lina-group
```

---

## Step 2 — Upload the training script to the PVC

The training jobs read `train_option2.py` from `/pvc/scripts/` inside the container.
You need to copy your local script there using the uploader pod.

### 2a. Start the uploader pod

```bash
kubectl apply -f k8s/script-uploader.yaml -n gai-lina-group
```

Wait for it to be Running:
```bash
kubectl get pod script-uploader -n gai-lina-group -w
```

Press Ctrl+C once you see `Running`.

### 2b. Create the scripts directory on the PVC

```bash
kubectl exec -n gai-lina-group script-uploader -- mkdir -p /pvc/scripts
```

### 2c. Copy the training script

```bash
kubectl cp k8s/train_option2.py gai-lina-group/script-uploader:/pvc/scripts/train_option2.py
```

Verify the copy:
```bash
kubectl exec -n gai-lina-group script-uploader -- ls -lh /pvc/scripts/
```

You should see `train_option2.py` listed.

### 2d. Create the checkpoints directory (if it does not exist yet)

```bash
kubectl exec -n gai-lina-group script-uploader -- mkdir -p /pvc/checkpoints
```

### 2e. Delete the uploader pod (optional, frees resources)

```bash
kubectl delete pod script-uploader -n gai-lina-group
```

---

## Step 3 — Submit the training jobs

You can run both models in parallel (each on its own GPU node) or one at a time.

### Submit Model 1 (T1 <-> T1Gd):
```bash
kubectl apply -f k8s/train-t1-t1gd.yaml -n gai-lina-group
```

### Submit Model 2 (T2 <-> FLAIR):
```bash
kubectl apply -f k8s/train-t2-flair.yaml -n gai-lina-group
```

Check that the jobs were created:
```bash
kubectl get jobs -n gai-lina-group
```

---

## Step 4 — Monitor the jobs

### Watch pod status:
```bash
kubectl get pods -n gai-lina-group -w
```

The pod name will be something like `fast-cwdm-t1-t1gd-xxxxx`.

### Stream live logs (Model 1):
```bash
kubectl logs -f -n gai-lina-group -l app=fast-cwdm,pair=t1-t1gd
```

### Stream live logs (Model 2):
```bash
kubectl logs -f -n gai-lina-group -l app=fast-cwdm,pair=t2-flair
```

### Get logs by pod name directly:
```bash
# First, get the exact pod name:
kubectl get pods -n gai-lina-group -l app=fast-cwdm

# Then:
kubectl logs -f -n gai-lina-group <pod-name>
```

Expected log output example:
```
Device : cuda
GPU    : Tesla V100-SXM2-16GB
VRAM   : 14.2 GB free / 16.0 GB total
Parameters : 18.54M
Pair=T1_T1Gd  Train=107  Val=13
Iter      100 | loss=0.0821 | lr=1.00e-05
Iter      200 | loss=0.0794 | lr=1.00e-05
...
Iter     5000 | loss=0.0612 | val=0.0589 | best=0.0589 | lr=9.99e-06
  Best model saved at iter 5000
```

---

## Step 5 — Check saved checkpoints

Use the uploader pod (or re-create it) to inspect what has been saved:

```bash
kubectl apply -f k8s/script-uploader.yaml -n gai-lina-group
kubectl exec -n gai-lina-group script-uploader -- ls -lh /pvc/checkpoints/
```

Expected files after training:
```
best_model_t1_t1gd.pt              # best val loss checkpoint, Model 1
checkpoint_t1_t1gd_iter0010000.pt  # periodic checkpoint at 10k iters
checkpoint_t1_t1gd_iter0020000.pt
...
best_model_t2_flair.pt
checkpoint_t2_flair_iter0010000.pt
...
```

---

## Step 6 — Resume training from a checkpoint

If a job fails or you want to extend training, resume from the latest checkpoint.

Edit the relevant YAML and add `--resume` to the python command. For example, in `train-t1-t1gd.yaml`:

```yaml
python /app/fast-cwdm/train_option2.py \
  --pair        T1_T1Gd \
  --iters       500000 \
  --lr          1e-5 \
  --num_channels 32 \
  --grad_accum  4 \
  --data_dir    /pvc/data/brats23/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
  --save_dir    /pvc/checkpoints \
  --repo_dir    /app/fast-cwdm \
  --log_every   100 \
  --val_every   5000 \
  --save_every  10000 \
  --num_workers 4 \
  --resume      /pvc/checkpoints/checkpoint_t1_t1gd_iter0100000.pt
```

Then delete the old job and resubmit:
```bash
kubectl delete job fast-cwdm-t1-t1gd -n gai-lina-group
kubectl apply -f k8s/train-t1-t1gd.yaml -n gai-lina-group
```

---

## Troubleshooting

### Job stays in Pending state
The scheduler cannot find a node matching the GPU affinity. Check available nodes:
```bash
kubectl get nodes -n gai-lina-group -o custom-columns=NAME:.metadata.name,GPU:.metadata.labels.'nvidia\.com/gpu\.product'
```

### OOMKilled (out of memory)
Full 224x224x160 volumes are large. If you see OOMKilled, increase the memory request/limit in the YAML (currently 48Gi request / 64Gi limit) and resubmit.

### Pod goes to Error immediately
Check what happened:
```bash
kubectl logs -n gai-lina-group <pod-name> --previous
```

### Script not found: /pvc/scripts/train_option2.py
The copy in Step 2 did not complete. Re-run Steps 2a through 2c.

### Check job status details:
```bash
kubectl describe job fast-cwdm-t1-t1gd -n gai-lina-group
kubectl describe pod <pod-name> -n gai-lina-group
```

---

## Quick Reference

| Command | Purpose |
|---|---|
| `kubectl apply -f <file> -n gai-lina-group` | Create/update a resource |
| `kubectl delete job <name> -n gai-lina-group` | Delete a job (required before resubmit) |
| `kubectl get pods -n gai-lina-group` | List all pods |
| `kubectl get jobs -n gai-lina-group` | List all jobs |
| `kubectl logs -f -n gai-lina-group <pod>` | Stream pod logs |
| `kubectl exec -n gai-lina-group <pod> -- <cmd>` | Run command inside pod |
| `kubectl cp <src> gai-lina-group/<pod>:<dst>` | Copy file into pod |
| `kubectl describe pod <pod> -n gai-lina-group` | Detailed pod info / events |
