# Landsat (RS) arm on Google Earth Engine

These scripts rebuild the remote-sensing baseline of Ma et al. (2026) as county means over CDL corn
pixels. They are needed only for predictions P1 and P6, which compare AlphaEarth overlap with
Landsat overlap. See [docs/methods.md §8](../docs/methods.md) for the method and the documented
deviations.

| Script | Purpose |
|---|---|
| `authenticate.py` | One-time Earth Engine sign-in. It uses the Python API because the `earthengine` CLI in earthengine-api 1.6.15 needs Python >= 3.10. |
| `export_landsat_harmonics.py` | Starts export tasks: per-pixel harmonic fits, averaged over corn pixels per county, written to Google Drive as CSV |
| `check_tasks.py` | Lists recent tasks with state and EECU-hours used |

```bash
.venv/bin/pip install earthengine-api==1.6.15
.venv/bin/python gee/authenticate.py
# 1. seconds, almost no compute: one county, printed and sanity-checked (GCVI peak timing and level)
.venv/bin/python gee/export_landsat_harmonics.py --project <gcp-project> --lite --years 2017 --smoke 19169
# 2. measure the cost of one year before committing to eight
.venv/bin/python gee/export_landsat_harmonics.py --project <gcp-project> --lite --years 2017
.venv/bin/python gee/check_tasks.py <gcp-project>
# 3. put the Drive CSVs in data/gee/, then
.venv/bin/python -m aeoverlap.rs_features && make preregistered evaluate figures
```

## Compute: read before running

On a noncommercial Earth Engine project, the 12-state `--lite` export for a single year used about
**25 EECU-hours per minute** and passed 136 EECU-hours within 6 minutes, with no measurable progress
(2026-11-07). The free Community tier provides 150 EECU-hours per month. Running all eight years
realistically needs the Contributor or Partner tier, or the source study's own RS features.

## Validation gate

Before RS results are used, the RS arm must reproduce Ma et al. (2026), Table 6: East->West R2 of
about 0.66 and West->East of about 0.60. For `--lite`, Amendment 1 of the pre-registration sets a
looser gate: within 0.15 of those values.
