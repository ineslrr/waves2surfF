#!/bin/bash
# Compute MFWAM regional mean/std for 2023.
#
# Usage:
#   bash compute_mfwam_norm_stats.sh
#   REGION=gulfstream bash compute_mfwam_norm_stats.sh
#   LON_MIN=10 LON_MAX=50 LAT_MIN=-45 LAT_MAX=-25 bash compute_mfwam_norm_stats.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATS_DIR="${STATS_DIR:-${SCRIPT_DIR}/../stats}"

DATA_DIR="${DATA_DIR:-/home/datawork-WW3/PROJECT/AMPHITRITE/BASELINES/MFWAM}"
START_DATE="${START_DATE:-20230101}"
END_DATE="${END_DATE:-20231231}"
REGION="${REGION:-agulhas}"
OUTPUT="${OUTPUT:-${STATS_DIR}/mfwam_${REGION}_${START_DATE}_${END_DATE}_stats.json}"
STRIDE_DAYS="${STRIDE_DAYS:-1}"

mkdir -p "${STATS_DIR}"

echo "=== MFWAM norm stats ==="
echo "Data   : ${DATA_DIR}"
echo "Period : ${START_DATE} -> ${END_DATE}"
echo "Region : ${REGION}"
echo "Output : ${OUTPUT}"
echo

# Prefer named region; override with explicit lon/lat if all four are set
EXTRA_ARGS=()
if [[ -n "${LON_MIN:-}" && -n "${LON_MAX:-}" && -n "${LAT_MIN:-}" && -n "${LAT_MAX:-}" ]]; then
    EXTRA_ARGS=(--lon_min "${LON_MIN}" --lon_max "${LON_MAX}" --lat_min "${LAT_MIN}" --lat_max "${LAT_MAX}")
    REGION_ARGS=()
else
    REGION_ARGS=(--region "${REGION}")
fi

python "${SCRIPT_DIR}/compute_mfwam_norm_stats.py" \
    --data_dir "${DATA_DIR}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output "${OUTPUT}" \
    --stride_days "${STRIDE_DAYS}" \
    "${REGION_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
