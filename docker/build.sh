#!/usr/bin/env bash
# Build the training image from the repo root.
# Run from the repo root:  bash docker/build.sh
REPO_ROOT="$( cd "$(dirname "$0")/.." ; pwd -P )"
docker build --platform linux/amd64 \
    -t ghcr.io/amalsha27/bidirectional-mri-synthesis:latest \
    -f "$REPO_ROOT/docker/Dockerfile.train" \
    "$REPO_ROOT"
