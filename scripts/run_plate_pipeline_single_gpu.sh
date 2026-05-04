#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/servidor/ArtigoLightglue}"
GPU_ID="${GPU_ID:-2}"
BATCH="${BATCH:-16}"
LOADER_WORKERS="${LOADER_WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/plate_pipeline_gpu${GPU_ID}_${RUN_ID}.log}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

exec > >(tee -a "${LOG_PATH}") 2>&1

echo "=== Plate pipeline single GPU ==="
echo "start: $(date --iso-8601=seconds)"
echo "root: ${ROOT}"
echo "gpu_id: ${GPU_ID}"
echo "device: ${DEVICE}"
echo "batch: ${BATCH}"
echo "loader_workers: ${LOADER_WORKERS}"
echo "log: ${LOG_PATH}"
echo

echo "=== Environment ==="
python3 --version
nvidia-smi
echo

echo "=== Step 1/3: detector + frame selection + symlinks ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 benchmarks/select_plate_frames_for_cvat.py \
  --device "${DEVICE}" \
  --batch "${BATCH}" \
  --loader-workers "${LOADER_WORKERS}" \
  --force

echo
echo "=== Step 2/3: create CVAT tasks in project 79 ==="
python3 benchmarks/create_cvat_tasks_from_selected_frames.py --force-create

echo
echo "=== Step 3/3: LightGlue tracker + upload annotations ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 benchmarks/run_lightglue_selected_frames_to_cvat.py \
  --device "${DEVICE}" \
  --upload

echo
echo "finished: $(date --iso-8601=seconds)"
echo "log: ${LOG_PATH}"
