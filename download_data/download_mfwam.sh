#!/bin/bash
# Companion launcher for download_mfwam.py (same folder).
#
# Usage:
#   bash download_mfwam.sh                  # global (default)
#   REGION=biscay bash download_mfwam.sh    # small test box
#   START_DATE=20240102 END_DATE=20240102 bash download_mfwam.sh
#   FORCE_CMEMS_LOGIN=1 bash download_mfwam.sh   # force re-login
#
# Prerequisites:
#   pip install copernicusmarine xarray netCDF4
#   one-time login is handled by this script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

START_DATE="${START_DATE:-20240101}"
END_DATE="${END_DATE:-20240101}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/temp/MFWAM_test}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${OUTPUT_DIR}/_tmp}"
REGION="${REGION:-global}"   # global | biscay
CMEMS_USERNAME="${CMEMS_USERNAME:-ssoares}"
FORCE_CMEMS_LOGIN="${FORCE_CMEMS_LOGIN:-0}"
LOGIN_STATE_FILE="${HOME}/.config/waves2surf/cmems_login_done"

if ! command -v python >/dev/null 2>&1; then
    echo "python not found in PATH. Activate the project environment first."
    exit 1
fi

if ! python -c "import copernicusmarine" >/dev/null 2>&1; then
    echo "copernicusmarine not available in this Python environment."
    echo "Install in your project env with: python -m pip install copernicusmarine xarray netCDF4"
    exit 1
fi

if ! command -v copernicusmarine >/dev/null 2>&1; then
    echo "copernicusmarine CLI not found in PATH."
    echo "Activate the correct project environment before running this script."
    exit 1
fi

if [ "${FORCE_CMEMS_LOGIN}" = "1" ] || [ ! -f "${LOGIN_STATE_FILE}" ]; then
    echo "Copernicus login required (first time on this machine for this project)."
    echo "Username: ${CMEMS_USERNAME}"
    echo "Password will be requested securely by the Copernicus CLI."

    # The CLI stores auth locally so future runs do not prompt for credentials.
    copernicusmarine login --username "${CMEMS_USERNAME}"

    mkdir -p "$(dirname "${LOGIN_STATE_FILE}")"
    touch "${LOGIN_STATE_FILE}"
fi

mkdir -p "${OUTPUT_DIR}" "${DOWNLOAD_DIR}"

BBOX_ARGS=()
if [ "${REGION}" = "biscay" ]; then
    BBOX_ARGS=(--lon_min -10 --lon_max 0 --lat_min 43 --lat_max 50)
elif [ "${REGION}" = "agulhas" ]; then
    BBOX_ARGS=(--lon_min 10 --lon_max 35 --lat_min -45 --lat_max -35)
elif [ "${REGION}" = "california_current" ]; then
    BBOX_ARGS=(--lon_min -145 --lon_max -120 --lat_min 30 --lat_max 40)
fi

echo "=== MFWAM download ==="
echo "Script : ${SCRIPT_DIR}/download_mfwam.py"
echo "Period : ${START_DATE} -> ${END_DATE}"
echo "Region : ${REGION}"
echo "Output : ${OUTPUT_DIR}"
echo

python "${SCRIPT_DIR}/download_mfwam.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    --download_dir "${DOWNLOAD_DIR}" \
    --variables VHM0 VTM10 VMDR \
    "${BBOX_ARGS[@]}" \
    --overwrite

echo
echo "=== Done. Check output with: ==="
echo "ls -lh ${OUTPUT_DIR}/*/*/*.nc"
echo "ncdump -h ${OUTPUT_DIR}/${START_DATE:0:4}/${START_DATE:4:2}/${START_DATE}.nc"
