# Overlap and transfer of AlphaEarth crop-yield models across Corn Belt states

A small, pre-registered test built on the county corn data shared by
Ma, Shen, Swatantran & Lobell (2026), *Harvesting AlphaEarth*, IJAEOG
([repo](https://github.com/yuchima8/Harvest_AlphaEarth)).

**Question.** Their Table 6 shows AlphaEarth (AEF) yield models transferring poorly between the
eastern and western Corn Belt, while Landsat (RS) models hold up, and their Figure 8 shows the two
regions' embeddings barely overlapping. With 12 states there are 132 ordered transfer pairs, not
two. That is enough to ask whether overlap, measured without any target yields, predicts how much
a model loses when moved, and whether thinner overlap is why AEF loses more.

The predictions, measures and tests were fixed in [PREREGISTRATION.md](PREREGISTRATION.md) before
any pair-level transfer loss was computed.

## Run

```bash
./get_data.sh                                   # downloads the paper's shared CSV into data/
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_units.py            # synthetic checks
.venv/bin/python src/run_all.py                 # E1 reproduction, E3 132-pair matrix, E4 overlap
.venv/bin/python src/run_tests.py               # pre-registered tests -> results/summary.md
.venv/bin/python src/make_figure.py             # results/figure.png
```

RS arm: `gee/export_landsat_harmonics.py` (Earth Engine) and then `src/rs_features.py`. Once
`data/rs_features.csv` exists, the scripts above add the `RS` and `RS-noclim` arms automatically.
**Gate:** the RS arm must reproduce Table 6 (East->West R2 ~0.66, West->East ~0.60) before its
results are used.

## Results

Pending.
