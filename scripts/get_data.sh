#!/usr/bin/env bash
# Fetch the county-level corn table shared by Ma, Shen, Swatantran & Lobell (2026) from
# github.com/yuchima8/Harvest_AlphaEarth, pinned to the upstream commit that last changed it,
# and verify its SHA-256. The upstream repository has no license file, so the data are not
# redistributed here.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=2ec9163b6105337a4b4fd26848d0df7f11a12fb5
SHA256=310f4982b21206e2f4dfe8773116eb1dea3be4b627b6d98b555aed12ca8a7e33
OUT=data/US_FM_Corn_Data_shared.csv
mkdir -p data
curl -sSfL -o "$OUT" "https://raw.githubusercontent.com/yuchima8/Harvest_AlphaEarth/${COMMIT}/code/US_FM_Corn_Data_shared.csv"
GOT=$( (command -v sha256sum >/dev/null && sha256sum "$OUT" || shasum -a 256 "$OUT") | cut -d' ' -f1)
if [ "$GOT" != "$SHA256" ]; then
  echo "checksum mismatch for $OUT: got $GOT, expected $SHA256" >&2
  exit 1
fi
echo "saved $OUT ($(wc -l < "$OUT" | tr -d ' ') lines, sha256 verified)"

# Exploratory follow-ups only: USGS county water use, 2015 (Dieter et al. 2018, doi:10.5066/F7TB15V5).
# U.S. Government work (public domain); fetched rather than committed to keep the repository small.
USGS_SHA256=b7980ea2fe27152a0674a4318f828d76ae0f238823f29990edcce32c3b440d50
USGS_OUT=data/usgs/usco2015v2.0.csv
mkdir -p data/usgs
curl -sSfL -o "$USGS_OUT" "https://www.sciencebase.gov/catalog/file/get/5af3311be4b0da30c1b245d8?f=__disk__eb%2F74%2Feb%2Feb74ebb41169c76aaf374990bd5a71cac82604c1"
GOT=$( (command -v sha256sum >/dev/null && sha256sum "$USGS_OUT" || shasum -a 256 "$USGS_OUT") | cut -d' ' -f1)
if [ "$GOT" != "$USGS_SHA256" ]; then
  echo "checksum mismatch for $USGS_OUT: got $GOT, expected $USGS_SHA256" >&2
  exit 1
fi
echo "saved $USGS_OUT (sha256 verified)"
