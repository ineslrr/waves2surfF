#!/bin/bash
# Compute Mercator surface-current regional mean/std for the training year.
#
# Usage:
#   bash compute_current_norm_stats.sh
#   REGION=agulhas bash compute_current_norm_stats.sh
#   STRIDE_HOURS=3 bash compute_current_norm_stats.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATS_DIR="${STATS_DIR:-${SCRIPT_DIR}/../stats}"

DATA_DIR="${DATA_DIR:-/home/datawork-WW3/FORCING/MERCA_GLOB_PHY_FOR/NC4}"
START_DATE="${START_DATE:-20230101}"
END_DATE="${END_DATE:-20231231}"
REGION="${REGION:-agulhas}"
STRIDE_HOURS="${STRIDE_HOURS:-1}"
DEPTH_INDEX="${DEPTH_INDEX:-0}"
OUTPUT="${OUTPUT:-${STATS_DIR}/mercator_${REGION}_${START_DATE}_${END_DATE}_stats.json}"

mkdir -p "${STATS_DIR}"

echo "=== Mercator current norm stats ==="
echo "Data   : ${DATA_DIR}"
echo "Period : ${START_DATE} -> ${END_DATE}"
echo "Region : ${REGION}"
echo "Stride : ${STRIDE_HOURS} h"
echo "Output : ${OUTPUT}"
echo

EXTRA_ARGS=()
if [[ -n "${LON_MIN:-}" && -n "${LON_MAX:-}" && -n "${LAT_MIN:-}" && -n "${LAT_MAX:-}" ]]; then
    EXTRA_ARGS=(--lon_min "${LON_MIN}" --lon_max "${LON_MAX}" --lat_min "${LAT_MIN}" --lat_max "${LAT_MAX}")
    REGION_ARGS=()
else
    REGION_ARGS=(--region "${REGION}")
fi

python "${SCRIPT_DIR}/compute_current_norm_stats.py" \
    --data_dir "${DATA_DIR}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output "${OUTPUT}" \
    --stride_hours "${STRIDE_HOURS}" \
    --depth_index "${DEPTH_INDEX}" \
    --variables uo vo \
    "${REGION_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
