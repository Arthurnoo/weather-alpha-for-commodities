# Weather Stress Index for commodity futures

ENSAE, *Machine Learning for Portfolio Management and Trading* (S. Champonnois).

Does weather in production areas (consumption areas for natural gas) help forecast commodity
futures, and does a synthetic Weather Stress Index beat the individual weather variables?

Universe: corn, soybeans, SRW wheat, HRW wheat, arabica coffee, cocoa, sugar, natural gas.
Sample: weekly, 2000 to 2025 (weather from 1990 to build past-only climatologies).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run order (from the project root)

```bash
.venv/bin/python -m src.download_prices   # Yahoo futures, a few seconds
.venv/bin/python -m src.download_spam     # MapSPAM crop production, ~150 MB
.venv/bin/python -m src.download_cpc      # NOAA CPC daily weather, slow (rerun until complete)
.venv/bin/python -m src.prices            # weekly returns, features, targets
.venv/bin/python -m src.weights           # production weights on the weather grid
.venv/bin/python -m src.weather           # weekly weather per commodity and zone
.venv/bin/python -m src.components        # stress anomalies
.venv/bin/python -m src.indices           # WSI-CPRI and WSI-PCA
```

Then open `notebooks/main.ipynb`: it only reads `data/processed/` and runs the models.

## Layout

| File | Role |
|---|---|
| `config.py` | universe, zones, seasons, thresholds, dates, paths |
| `src/download_*.py` | raw downloads into `data/raw/` |
| `src/prices.py` | roll-adjusted weekly returns, market features, targets |
| `src/weights.py` | production (or population) weights per weather cell |
| `src/weather.py` | daily stress variables, weighted average, weekly aggregation |
| `src/components.py` | past-only anomalies, season mask, zones combined |
| `src/indices.py` | CPRI-style and PCA indices |
| `src/models.py` | walk-forward M1 to M4, ridge and boosting |
| `notebooks/main.ipynb` | all results and figures |

`data/` is not versioned: rerun the scripts to rebuild it.
