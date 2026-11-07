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
