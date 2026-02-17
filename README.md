# priprava_podatkov_copy

Projekt uporablja en eksperimentalni runner za primerjavo več arhitektur napovedi (1-step ahead):

- arhitekture: `transformer`, `lstm`, `gru`, `hybrid_loclstm`
- pristop: za vsak vnos v `data.stations` se nauči ločen model (in ločen run)

Opomba: `hybrid_loclstm` uporablja station embedding in opcijski geo vektor (`data.station_geo_path`), ostale arhitekture identiteto postaje obravnavajo kot navadno vhodno značilko (stolpec `station` v `data.features`).

Vire in obseg učenja upravljaš prek `data.data_dir` + `data.stations`:
- `data.stations: [all_stations]` -> uporabi `data/raw/all_stations.csv` (vse postaje),
- `data.stations: [E403]` -> uporabi `data/raw/E403.csv` (ena postaja),
- `data.stations: [E403, E404, E405, all_stations]` -> za vsak vir zažene svoj trening; če manjka stolpec `station`, se infera iz imena datoteke.

## Eksperimenti

Runner podpira oba scenarija:

1. `standard`
- vse postaje imajo poln učni interval,
- treniranje na prvih `experiment.standard.train_ratio` deležu časovne osi,
- evalvacija na preostanku podatkov (del > 1 leto).

2. `five_fold`
- 5-fold delitev po postajah,
- v vsakem fold-u je 20% postaj označenih kot `reduced` (KFold),
- za te postaje se uporablja manjši train delež `experiment.five_fold.reduced_train_ratio`,
- evalvacija ostane enaka kot pri `standard`.

### Natančen protokol (low-data stress-test)

Ta `five_fold` ni klasični cross-validation z "train na 4, test na 1". Namen je simulacija postaj z manj podatki:

1. V iteraciji `k` izberemo en fold postaj kot `reduced` (~20% postaj).
2. `reduced` postajam uporabimo krajši train del glede na `reduced_train_ratio`.
3. Ostale postaje (~80%) imajo poln train interval.
4. Za vse vključene arhitekture (`experiment.common.models`) učimo en model od začetka v vsaki iteraciji.
5. Evalvacija ostane časovna in enaka kot v `standard` scenariju (preostanek po prvem letu).

## Metrike

Za vse modele in vse fold-e se računajo:

- `MAE`
- `MAPE` (z nastavljivim floor-om imenovalca prek `metrics.mape_min_abs_target`)
- `MSE`
- `RMSE`
- `EVS`
- `R2`
- `train_time_sec`
- dodatno tudi `sMAPE` in `WAPE`

## Glavne datoteke

- modeli: `src/air_quality_imputer/models/transformer_forecaster.py`, `src/air_quality_imputer/models/recurrent_forecasters.py`
- runner: `src/air_quality_imputer/training/forecast_runner.py`
- DVC pipeline: `dvc.yaml`
- parametri (single source of truth): `configs/pipeline/params.yaml`
- v `models.<ime_modela>` ima vsak model svoj `runtime` (optimizer/compile/dataloader) in `params` (arhitektura)
- modeli ne delajo več implicitnega NaN fill-a; podatki morajo biti brez `NaN/Inf`

## Zagon

### DVC (priporočeno)

```bash
dvc repro
```

Eksperimenti/override-i prek DVC:

```bash
dvc exp run \
  -S configs/pipeline/params.yaml:experiment.five_fold.run=false \
  -S configs/pipeline/params.yaml:training.epochs=2 \
  -S configs/pipeline/params.yaml:output.save_models=false

dvc exp show
dvc metrics show
```

Metrike za DVC so v `reports/metrics/forecast_metrics.json`.

### Runner (brez DVC)

Po installu paketa:

```bash
aqi-run-experiments --params configs/pipeline/params.yaml
```

Privzeto zažene samo `standard`. Če želiš tudi `five_fold`, nastavi `experiment.five_fold.run=true`.

Brez installa:

```bash
PYTHONPATH=src python -m air_quality_imputer.training.forecast_runner --params configs/pipeline/params.yaml
```

Primer za hiter smoke config (kopija YAML + zagon):

```bash
cp configs/pipeline/params.yaml /tmp/aqi_smoke.yaml
# nato uredi /tmp/aqi_smoke.yaml (npr. training.epochs=1)
PYTHONPATH=src python -m air_quality_imputer.training.forecast_runner --params /tmp/aqi_smoke.yaml
```

## Izhodi

Rezultati se shranijo v `reports/forecast_experiments/`:

- `summary_metrics.csv`: skupni rezultati po modelu/fold-u/scenariju
- `station_metrics.csv`: metrike po postajah
- `fold_assignments.csv`: katere postaje so bile `reduced` v posameznem fold-u
- `resolved_config.yaml`: dejanska konfiguracija iz podanega `--params` YAML
