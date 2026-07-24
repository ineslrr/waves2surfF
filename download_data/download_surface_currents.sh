#!/bin/bash
# Companion launcher for download_surface_currents.py (same folder).
# Product: GLOBAL_ANALYSISFORECAST_PHY_001_024 (Mercator uo/vo).
#
# Usage:
#   bash download_surface_currents.sh                       # global, 6-hourly
#   REGION=biscay bash download_surface_currents.sh         # small test box
#   FREQ=daily bash download_surface_currents.sh            # daily-mean currents
#   START_DATE=20240102 END_DATE=20240102 bash download_surface_currents.sh
#
# Prerequisites:
#   pip install copernicusmarine xarray netCDF4
#   copernicusmarine login

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

START_DATE="${START_DATE:-20240101}"
END_DATE="${END_DATE:-20240101}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/temp/CUR_test}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${OUTPUT_DIR}/_tmp}"
REGION="${REGION:-global}"   # global | biscay | agulhas | california_current
FREQ="${FREQ:-6hourly}"      # 6hourly | daily

mkdir -p "${OUTPUT_DIR}" "${DOWNLOAD_DIR}"

BBOX_ARGS=()
if [ "${REGION}" = "biscay" ]; then
    BBOX_ARGS=(--lon_min -10 --lon_max 0 --lat_min 43 --lat_max 50)
elif [ "${REGION}" = "agulhas" ]; then
    BBOX_ARGS=(--lon_min 10 --lon_max 35 --lat_min -45 --lat_max -35)
elif [ "${REGION}" = "california_current" ]; then
    BBOX_ARGS=(--lon_min -145 --lon_max -120 --lat_min 30 --lat_max 40)
fi

echo "=== Surface currents download (PHY_001_024) ==="
echo "Script : ${SCRIPT_DIR}/download_surface_currents.py"
echo "Period : ${START_DATE} -> ${END_DATE}"
echo "Region : ${REGION}"
echo "Freq   : ${FREQ}"
echo "Output : ${OUTPUT_DIR}"
echo

python "${SCRIPT_DIR}/download_surface_currents.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    --download_dir "${DOWNLOAD_DIR}" \
    --freq "${FREQ}" \
    --variables uo vo \
    "${BBOX_ARGS[@]}" \
    --overwrite

echo
echo "=== Done. Check output with: ==="
echo "ls -lh ${OUTPUT_DIR}/*/*/*.nc"
echo "ncdump -h ${OUTPUT_DIR}/${START_DATE:0:4}/${START_DATE:4:2}/${START_DATE}.nc"
