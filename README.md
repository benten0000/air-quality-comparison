# priprava_podatkov_copy

Projekt uporablja en eksperimentalni runner za primerjavo več arhitektur in dveh pristopov napovedi (1-step ahead):

- arhitekture: `transformer`, `lstm`, `gru`, `hybrid_loclstm`
- pristopa: `global_*` (en model za vse postaje) in `per_station_*` (ločen model na postajo)

Opomba: `hybrid_loclstm` uporablja station embedding in opcijski geo vektor (`data.station_geo_path`), ostale arhitekture identiteto postaje obravnavajo kot navadno vhodno značilko (stolpec `station` v `data.features`).

## Eksperimenti

Runner podpira oba scenarija:

1. `standard`
- vse postaje imajo poln učni interval,
- treniranje na prvih `train_months` mesecih,
- evalvacija na preostanku podatkov (del > 1 leto).

2. `five_fold`
- 5-fold delitev po postajah,
- v vsakem fold-u je 20% postaj označenih kot `reduced` (KFold),
- za te postaje se pri učenju odstrani prvih `reduced_months` (privzeto 6 mesecev),
- evalvacija ostane enaka kot pri `standard`.

### Natančen protokol (low-data stress-test)

Ta `five_fold` ni klasični cross-validation z "train na 4, test na 1". Namen je simulacija postaj z manj podatki:

1. V iteraciji `k` izberemo en fold postaj kot `reduced` (~20% postaj).
2. `reduced` postajam v train obdobju odrežemo prvih `reduced_months` (privzeto 6 mesecev).
3. Ostale postaje (~80%) imajo poln train interval.
4. Za vse vključene arhitekture (`experiment.models`) učimo oba pristopa (`global_*`, `per_station_*`) od začetka v vsaki iteraciji.
5. Evalvacija ostane časovna in enaka kot v `standard` scenariju (preostanek po prvem letu).

Če je vključeno `experiment.run_five_fold_if_global_not_worse=true`, se `five_fold` izvede samo, če je globalni pristop na `standard` scenariju primerljiv ali boljši od per-station glede na izbran `experiment.gate_metric` (npr. `rmse`) in `experiment.gate_tolerance_pct`.

## Metrike

Za oba pristopa in vse fold-e se računajo:

- `MAE`
- `MAPE` (z nastavljivim floor-om imenovalca prek `metrics.mape_min_abs_target`)
- `MSE`
- `RMSE`
- `EVS`
- `R2`
- `train_time_sec`
- dodatno tudi `sMAPE` in `WAPE`

## Glavne datoteke

- modeli: `src/air_quality_imputer/models/transformer_imputer.py`, `src/air_quality_imputer/models/recurrent_forecasters.py`
- runner: `src/air_quality_imputer/training/forecast_runner.py`
- DVC pipeline: `dvc.yaml`
- parametri (DVC): `configs/pipeline/params.yaml`
- konfiguracija (Hydra): `conf/forecast.yaml` (naj ostane usklajena z `configs/pipeline/params.yaml`)

## Zagon

### DVC (priporočeno)

```bash
dvc repro
```

Eksperimenti/override-i prek DVC:

```bash
dvc exp run \
  -S configs/pipeline/params.yaml:experiment.run_five_fold=false \
  -S configs/pipeline/params.yaml:training.epochs=2 \
  -S configs/pipeline/params.yaml:output.save_models=false

dvc exp show
dvc metrics show
```

Metrike za DVC so v `reports/metrics/forecast_metrics.json`.

### Runner (brez DVC)

Po installu paketa:

```bash
aqi-run-experiments
```

Privzeto zažene samo `standard`. `five_fold` se izvede samo, če nastaviš `experiment.run_five_fold=true`.

Brez installa:

```bash
PYTHONPATH=src python -m air_quality_imputer.training.forecast_runner
```

Primeri override-ov:

```bash
# Samo standard eksperiment
PYTHONPATH=src python -m air_quality_imputer.training.forecast_runner \
  experiment.run_five_fold=false

# Hitri smoke run
PYTHONPATH=src python -m air_quality_imputer.training.forecast_runner \
  experiment.run_five_fold=false \
  training.epochs=2 \
  output.save_models=false

# Celoten protokol na manjšem podnaboru postaj (smoke za standard + five_fold)
PYTHONPATH=src python -m air_quality_imputer.training.forecast_runner \
  data.station_glob='E40*.csv' \
  training.epochs=1 \
  output.save_models=false
```

## Izhodi

Rezultati se shranijo v `reports/forecast_experiments/`:

- `summary_metrics.csv`: skupni rezultati po pristopu/fold-u/scenariju
- `station_metrics.csv`: metrike po postajah
- `fold_assignments.csv`: katere postaje so bile `reduced` v posameznem fold-u
- `resolved_config.yaml`: dejanska konfiguracija z override-i
