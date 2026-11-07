#!/usr/bin/env bash
# Downloads the county-level corn table shared by Ma, Shen, Swatantran & Lobell (2026)
# from github.com/yuchima8/Harvest_AlphaEarth. Not redistributed here.
set -euo pipefail
mkdir -p data
curl -sSL -o data/US_FM_Corn_Data_shared.csv \
  https://raw.githubusercontent.com/yuchima8/Harvest_AlphaEarth/main/code/US_FM_Corn_Data_shared.csv
echo "saved data/US_FM_Corn_Data_shared.csv ($(wc -l < data/US_FM_Corn_Data_shared.csv) lines)"
