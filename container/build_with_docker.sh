#!/usr/bin/env bash
# Build the Apptainer image without a local Apptainer installation (e.g. on macOS)
# by running Apptainer inside a privileged Docker container. The image is built
# for x86-64 Linux, the usual cluster architecture.
#
# Usage (from anywhere):  container/build_with_docker.sh [output.sif]
# The output path is relative to the repository root (default: opn.sif).
set -euo pipefail

APPTAINER_VERSION=1.5.3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-opn.sif}"

docker run --rm --privileged --platform linux/amd64 \
    -v "$ROOT":/repo -w /repo debian:bookworm-slim bash -c "
        set -e
        apt-get update -qq
        apt-get install -y -qq curl ca-certificates >/dev/null
        curl -fsSL -o /tmp/apptainer.deb \
            https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}_amd64.deb
        apt-get install -y -qq /tmp/apptainer.deb >/dev/null
        apptainer build --force '${OUT}' container/opn.def
    "
echo "Built ${ROOT}/${OUT}"
